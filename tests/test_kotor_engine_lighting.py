"""KOTOR engine light model (static RE of swkotor2.exe 817494) and the KOTOR-linear bake mode.

Evidence for every constant: astra/research/re/RE02_v001.md and
astra/research/vanilla_relight/engine_lighting_model_v001.md (KotorMods research tree).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.adapters.gpu.lightmap_baker import LightmapBaker
from src.adapters.gpu.lightmap_gpu_solver import LightmapGpuSolver
from src.core.geometry.model_data import KotorModel, ModelNode, NodeFlags
from src.core.lighting import kotor_engine_lighting as kel
from src.core.lighting.lightmap_bake_job import LightmapBakeJob
from src.core.lighting.lightmap_bake_settings import LightmapBakeSettings, uses_kotor_engine_model
from src.core.lighting.lightmap_lighting_solver import LightmapLightingSolver
from src.core.lighting.lightmap_rasterizer import LightmapRasterizer


def _floor_model() -> tuple[KotorModel, ModelNode, ModelNode]:
    model = KotorModel(name="tstlit")
    root = ModelNode(name="root", flags=int(NodeFlags.HEADER))
    mesh = ModelNode(
        name="floor01",
        flags=int(NodeFlags.HEADER | NodeFlags.MESH),
        parent=root,
        vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        uvs_lm=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        faces=[(0, 1, 2)],
        texture="tst_floor",
        lightmap="tst_floor_lm",
        has_lightmap=True,
        diffuse=(1.0, 1.0, 1.0),
    )
    root.children.append(mesh)
    model.root_node = root
    return model, root, mesh


def _point(name="AuroraLight01", position=(0.25, 0.25, 1.0), color=(0.8, 0.6, 0.4), radius=2.0, **extra):
    values = dict(
        name=name,
        source_type="Aurora",
        enabled=True,
        visible=True,
        type="aurora_point",
        position=position,
        color=color,
        intensity=1.0,
        radius=radius,
        casts_shadows=False,
        affects_lightmap=True,
    )
    values.update(extra)
    return SimpleNamespace(**values)


def _expected_point(positions, light, *, cutoff=1.0, multiplier=1.0) -> np.ndarray:
    delta = np.asarray(light.position, dtype=np.float64) - positions
    dist = np.linalg.norm(delta, axis=1)
    ndotl = np.maximum(delta[:, 2] / dist, 0.0)  # floor normal is +z
    att = 1.0 / (1.0 + 4.5 * dist**2 / light.radius**2)
    if cutoff > 0:
        att = np.where(dist <= cutoff * light.radius, att, 0.0)
    return np.clip((att * ndotl)[:, None] * np.asarray(light.color, dtype=np.float64) * multiplier, 0.0, 1.0)


# -- engine model ------------------------------------------------------------------------------------------------


def test_engine_attenuation_is_the_vertex_program_curve() -> None:
    r = 4.0
    d = np.asarray([0.0, 1.0, 4.0, 6.0])
    # env (1, 0, 9/(2 r^2), 1) with DST/DP3/RCP: 1 / (1 + 4.5 d^2 / r^2), 0.18 at d = r, no cut in the program.
    expected = 1.0 / (1.0 + 9.0 / (2.0 * r * r) * d**2)
    assert np.allclose(kel.engine_attenuation(d, r), expected)
    assert kel.engine_attenuation(r, r) == pytest.approx(1.0 / 5.5)
    assert np.allclose(kel.engine_attenuation(d, r, cutoff_scale=1.0), np.where(d <= r, expected, 0.0))
    # Fixed-function writer 0x00508350 clamps r <= 0.1 to 0.1; the vertex program does not.
    assert kel.engine_attenuation(0.1, 0.05, fixed_function=True) == pytest.approx(1.0 / 5.5)
    assert kel.engine_attenuation(0.1, 0.05) == pytest.approx(1.0 / 19.0)
    assert np.all(kel.engine_attenuation(d, 0.0) == 0.0)


def test_falloff_families_share_units_and_reject_unknown_names() -> None:
    d = np.asarray([0.0, 1.0, 2.0, 3.0])
    assert np.allclose(kel.falloff("engine", d, 2.0, cutoff_scale=0.0), 1.0 / (1.0 + 4.5 * (d / 2.0) ** 2))
    assert np.allclose(kel.falloff("linear", d, 2.0), [1.0, 0.5, 0.0, 0.0])
    assert np.allclose(kel.falloff("ghost", d, 2.0), [1.0, 0.25, 0.0, 0.0])
    assert kel.normalize_falloff_family("KOTOR") == "engine"
    with pytest.raises(ValueError):
        kel.falloff("inverse_square", d, 2.0)


def test_light_term_ambient_only_is_attenuated_without_ndotl_and_multiplier_is_ignored() -> None:
    position = np.asarray([[0.0, 0.0, 0.0]])
    facing_away = np.asarray([[0.0, 0.0, -1.0]])
    light = kel.KotorEngineLight(position=(0.0, 0.0, 2.0), color=(0.5, 0.25, 1.0), radius=4.0, multiplier=3.0)
    att = 1.0 / (1.0 + 4.5 * (2.0 / 4.0) ** 2)

    assert np.allclose(kel.light_term(position, facing_away, light), 0.0)
    ambient = kel.KotorEngineLight(position=(0.0, 0.0, 2.0), color=(0.5, 0.25, 1.0), radius=4.0, ambient_only=True, multiplier=3.0)
    assert np.allclose(kel.light_term(position, facing_away, ambient), np.asarray([[0.5, 0.25, 1.0]]) * att)
    up = np.asarray([[0.0, 0.0, 1.0]])
    assert np.allclose(kel.light_term(position, up, light), np.asarray([[0.5, 0.25, 1.0]]) * att)
    assert np.allclose(kel.light_term(position, up, light, apply_multiplier=True), np.asarray([[1.5, 0.75, 3.0]]) * att)


def test_object_light_selection_follows_the_engine_comparator() -> None:
    def light(name, x, *, dynamic_type=2, affect=False, priority=3, color=(1.0, 1.0, 1.0), radius=10.0):
        return kel.KotorEngineLight((x, 0.0, 0.0), color, radius, dynamic_type=dynamic_type, affect_dynamic=affect, priority=priority, name=name)

    lights = [
        light("far_type2", 9.0),
        light("near_type2", 1.0),
        light("type1", 5.0, dynamic_type=1),
        light("priority1", 8.0, priority=1),
        light("baked_only", 0.5, dynamic_type=0, affect=False),  # neither dynamic nor affect-dynamic
        light("affect_dynamic_static", 2.0, dynamic_type=0, affect=True),
        light("out_of_range", 30.0),
        light("black", 0.2, color=(0.0, 0.0, 0.0)),
    ]
    picked = kel.select_object_lights(lights, (0.0, 0.0, 0.0), limit=10)
    assert [item.name for item in picked] == ["priority1", "type1", "near_type2", "far_type2", "affect_dynamic_static"]
    assert [item.name for item in kel.select_object_lights(lights, (0.0, 0.0, 0.0))] == ["priority1", "type1", "near_type2"]


def test_static_rooms_take_type1_lights_only_without_a_radius_cut() -> None:
    lights = [
        kel.KotorEngineLight((50.0, 0.0, 0.0), (1.0, 0.0, 0.0), 5.0, dynamic_type=1, priority=2, name="saber"),
        kel.KotorEngineLight((1.0, 0.0, 0.0), (1.0, 1.0, 1.0), 5.0, dynamic_type=2, name="room_light"),
        kel.sun_light((0.2, 0.2, 0.2)),
    ]
    picked = kel.select_static_room_lights(lights, (0.0, 0.0, 0.0))
    assert [item.name for item in picked] == ["gidy_sun_diff", "saber"]
    assert kel.sun_light((1.0, 1.0, 1.0)).position == (4000.0, 4500.0, 7000.0)


def test_vertex_light_v0_adds_dyn_ambient_only_for_owned_parts_and_clamps() -> None:
    p = np.zeros((1, 3))
    n = np.asarray([[0.0, 0.0, 1.0]])
    kw = dict(self_illumination=(0.1, 0.0, 0.0), material_ambient=(0.5, 1.0, 1.0), scene_ambient=(0.4, 0.2, 0.0))
    assert np.allclose(kel.vertex_light_v0(p, n, [], owned=False, **kw), [[0.1, 0.0, 0.0]])
    assert np.allclose(kel.vertex_light_v0(p, n, [], owned=True, **kw), [[0.3, 0.2, 0.0]])
    hot = [kel.KotorEngineLight((0.0, 0.0, 0.5), (2.0, 2.0, 2.0), 10.0, dynamic_type=1)] * 5
    assert np.allclose(kel.vertex_light_v0(p, n, hot), 1.0)


def test_composite_rules_and_dc1_predictions() -> None:
    # Program 0x00990EC8: D * (v0 + LM).  DC-1 fixture: D = 0.5, LM 0.25/0.5/0.75, DynAmbient 0.25 on a gob room.
    table = kel.dc1_predictions()
    assert [row["static_room_additive"] for row in table["rows"]] == pytest.approx([0.125, 0.25, 0.375])
    assert [row["static_room_x2"] for row in table["rows"]] == pytest.approx([0.25, 0.5, 0.75])
    assert [row["gob_room_additive"] - row["static_room_additive"] for row in table["rows"]] == pytest.approx([0.125] * 3)
    assert kel.composite_lightmapped(0.8, 0.9, 0.5) == pytest.approx(1.0)
    # Env + lightmap (0x00990020): the lit term is not weighted by D.a and env is added, not mixed.
    env = kel.composite_lightmapped_env([0.5, 0.5, 0.5, 0.25], [0.4, 0.4, 0.4], [0.2, 0.2, 0.2], [0.4, 0.0, 0.0])
    assert np.allclose(env, [0.5 * 0.2 + 0.5 * 0.4 + 0.4 * 0.75, 0.3, 0.3])
    assert kel.lightmap_for_target(0.6, rule="additive", v0_rgb=0.2) == pytest.approx(0.4)
    assert kel.lightmap_for_target(0.6, rule="x2") == pytest.approx(0.3)
    with pytest.raises(ValueError):
        kel.composite_lightmapped(0.5, 0.5, rule="multiply")


# -- settings -------------------------------------------------------------------------------------------------------


def test_settings_validate_the_kotor_fields_without_throwing() -> None:
    settings = LightmapBakeSettings(
        lighting_model="photoreal",
        kotor_falloff="cubic",
        kotor_cutoff_scale=-1.0,
        encode_gamma=float("nan"),
        exposure=float("inf"),
    ).normalized()
    assert settings.lighting_model == "ghost"
    assert settings.kotor_falloff == "engine"
    assert settings.kotor_cutoff_scale == 1.0
    assert settings.encode_gamma == 1.0
    assert settings.exposure == 1.0
    assert len(settings.warnings) == 5

    kotor = LightmapBakeSettings.for_kotor_engine(kotor_cutoff_scale=2.0)
    assert uses_kotor_engine_model(kotor) and not uses_kotor_engine_model(LightmapBakeSettings())
    assert (kotor.include_diffuse, kotor.include_ambient, kotor.use_indirect_approximation) == (False, False, False)
    assert (kotor.encode_gamma, kotor.kotor_cutoff_scale, kotor.kotor_apply_multiplier) == (1.0, 2.0, False)
    assert LightmapBakeSettings(lighting_model="KOTOR").normalized().lighting_model == "kotor_engine"


# -- KOTOR-linear bake -----------------------------------------------------------------------------------------------


def test_kotor_bake_is_the_engine_light_term_in_stored_byte_units() -> None:
    _model, _root, mesh = _floor_model()
    buffer = LightmapRasterizer().rasterize_mesh(mesh, 1, 16)
    ys, xs = np.nonzero(buffer.valid_mask)
    positions = buffer.world_positions[ys, xs].astype(np.float64)
    light = _point()
    settings = LightmapBakeSettings.for_kotor_engine()
    solver = LightmapLightingSolver()

    baked = solver.solve_buffer(buffer, [light], settings)[ys, xs]
    assert np.allclose(baked, _expected_point(positions, light), atol=1.0e-6)
    assert "KOTOR engine light model" in solver.last_info and solver.last_warning == ""

    # Linear in the light colour (no tone map, no gamma) and independent of the material diffuse.
    doubled = solver.solve_buffer(buffer, [_point(color=(0.4, 0.3, 0.2)), _point(name="b", color=(0.4, 0.3, 0.2))], settings)
    assert np.allclose(doubled[ys, xs], baked, atol=1.0e-6)
    buffer.base_diffuse[...] = 0.25
    assert np.allclose(solver.solve_buffer(buffer, [light], settings)[ys, xs], baked, atol=1.0e-6)

    # The cut follows kotor_cutoff_scale; 0 disables it like the vertex program.
    short = _point(radius=1.1)
    cut = solver.solve_buffer(buffer, [short], settings)[ys, xs]
    uncut = solver.solve_buffer(buffer, [short], LightmapBakeSettings.for_kotor_engine(kotor_cutoff_scale=0.0))[ys, xs]
    assert np.allclose(cut, _expected_point(positions, short, cutoff=1.0), atol=1.0e-6)
    assert np.allclose(uncut, _expected_point(positions, short, cutoff=0.0), atol=1.0e-6)
    assert np.any(uncut > cut)

    # The scalar path agrees with the vectorised one.
    scalar = solver._solve_buffer_scalar(buffer, [light], settings)[ys, xs]
    assert np.allclose(scalar, baked, atol=1.0e-6)


def test_kotor_bake_keeps_negative_lights_and_ignores_the_multiplier_by_default() -> None:
    _model, _root, mesh = _floor_model()
    buffer = LightmapRasterizer().rasterize_mesh(mesh, 1, 16)
    ys, xs = np.nonzero(buffer.valid_mask)
    positions = buffer.world_positions[ys, xs].astype(np.float64)
    bright = _point(color=(0.9, 0.9, 0.9), intensity=4.0)
    dark = _point(name="neg", color=(-0.2, -0.2, -0.2))
    solver = LightmapLightingSolver()

    baked = solver.solve_buffer(buffer, [bright, dark], LightmapBakeSettings.for_kotor_engine())[ys, xs]
    expected = np.clip(_expected_point(positions, _point(color=(0.7, 0.7, 0.7))), 0.0, 1.0)
    assert np.allclose(baked, expected, atol=1.0e-6)

    applied = solver.solve_buffer(buffer, [bright], LightmapBakeSettings.for_kotor_engine(kotor_apply_multiplier=True))[ys, xs]
    assert np.allclose(applied, _expected_point(positions, bright, multiplier=4.0), atol=1.0e-6)


def test_kotor_bake_uses_the_light_world_position_like_the_engine() -> None:
    model, root, mesh = _floor_model()
    holder = ModelNode(name="lamp_holder", flags=int(NodeFlags.HEADER), parent=root, position=(0.0, 0.0, 0.5))
    lamp = ModelNode(name="AuroraLight07", flags=int(NodeFlags.HEADER | NodeFlags.LIGHT), parent=holder, position=(0.25, 0.25, 0.5))
    lamp.light_color = (0.8, 0.6, 0.4)
    lamp.light_radius = 2.0
    lamp.light_shadow = False
    holder.children.append(lamp)
    root.children.append(holder)
    settings = LightmapBakeSettings.for_kotor_engine()
    buffer = LightmapRasterizer().rasterize_mesh(mesh, 1, 16)
    ys, xs = np.nonzero(buffer.valid_mask)

    aurora = LightmapBaker().aurora_adapter.from_model(model)
    assert [light.name for light in aurora] == ["AuroraLight07"] and aurora[0].position == (0.25, 0.25, 0.5)
    from_node = LightmapLightingSolver().solve_buffer(buffer, aurora, settings)[ys, xs]
    at_world = LightmapLightingSolver().solve_buffer(buffer, [_point(position=(0.25, 0.25, 1.0))], settings)[ys, xs]
    assert np.allclose(from_node, at_world, atol=1.0e-6)


def test_kotor_bake_through_the_gpu_default_baker_writes_stored_bytes(tmp_path: Path) -> None:
    from PIL import Image

    model, _root, mesh = _floor_model()
    spot = _point(name="spot01", type="spot", cone_angle=170.0, direction=(0.0, 0.0, -1.0), color=(0.2, 0.2, 0.2))
    saber = _point(name="saber01", color=(0.1, 0.0, 0.0), dynamic_type=1)
    settings = LightmapBakeSettings.for_kotor_engine(
        use_gpu_acceleration=True,  # still evaluated on the CPU in float64
        resolution=64,
        output_directory=str(tmp_path),
        bake_visible_only=False,
        include_diffuse=True,  # ignored with a warning
    )
    result = LightmapBaker().bake(LightmapBakeJob(model=model, lights=[_point(), spot, saber], settings=settings))
    assert result.ok, result.errors
    assert any("KOTOR engine light model" in message for message in result.messages)
    joined = " ".join(result.warnings)
    assert "include_diffuse is ignored" in joined and "spot01" in joined and "saber01" in joined

    buffer = LightmapRasterizer().rasterize_mesh(mesh, 1, 64)
    ys, xs = np.nonzero(buffer.valid_mask)
    positions = buffer.world_positions[ys, xs].astype(np.float64)
    # spot parameters are never set by the engine: the spot bakes as a point light.
    expected = _expected_point(positions, _point()) + _expected_point(positions, spot) + _expected_point(positions, saber)
    expected_bytes = (np.clip(expected, 0.0, 1.0).astype(np.float32) * 255.0 + 0.5).astype(np.int64)
    stored = np.asarray(Image.open(result.bakes[0].output_path).convert("RGB"), dtype=np.int64)[ys, xs]
    # Stored bytes are the quantised engine term (float32 rounding may move a value sitting on a .5 boundary).
    assert np.max(np.abs(stored - expected_bytes)) <= 1
    assert np.mean(stored == expected_bytes) > 0.99
    manifest_settings = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))["settings"]
    assert manifest_settings["lighting_model"] == "kotor_engine"


def test_gpu_legacy_solve_matches_cpu_for_strengths_falloff_area_and_negative_lights() -> None:
    _model, _root, mesh = _floor_model()
    mesh.diffuse = (0.6, 0.7, 0.8)
    buffer = LightmapRasterizer().rasterize_mesh(mesh, 1, 16)
    lights = [
        _point(intensity=2.0, radius=1.5),
        _point(name="area01", type="area", area_size=4.0, position=(0.6, 0.1, 0.8), radius=2.5),
        _point(name="amb01", type="aurora_ambient", ambient_only=True, color=(0.3, 0.3, 0.5)),
        _point(name="neg01", color=(-0.6, -0.6, -0.6), position=(0.1, 0.6, 0.7)),
        _point(name="spot01", type="spot", cone_angle=90.0, direction=(0.0, 0.0, -1.0), position=(0.3, 0.3, 0.9)),
    ]
    settings = LightmapBakeSettings(
        include_ambient=True,
        ambient_strength=0.5,
        include_diffuse=True,
        diffuse_strength=0.5,
        use_indirect_approximation=True,
        indirect_strength=0.5,
        light_falloff_multiplier=1.5,
        use_shadows=False,
        gamma=1.0,  # keep float32 differences linear near zero
    ).normalized()
    gpu = LightmapGpuSolver()
    gpu_rgb = gpu.solve_buffer(buffer, lights, settings)
    if not gpu.last_info.startswith("GPU direct-light solve used"):
        pytest.skip(f"no ModernGL context: {gpu.last_info} {gpu.last_warning}")
    cpu_rgb = LightmapLightingSolver().solve_buffer(buffer, lights, settings)
    assert np.allclose(gpu_rgb, cpu_rgb, atol=1.0e-5)


def test_binary_mdl_light_colour_keeps_its_sign() -> None:
    from src.core.game.kotor_loader import _apply_bind_pose

    model, root, _mesh = _floor_model()
    lamp = ModelNode(name="AuroraLight09", flags=int(NodeFlags.HEADER | NodeFlags.LIGHT), parent=root)
    lamp.controllers = [{"type": 76, "values": [[-0.5, 0.2, 0.0]]}, {"type": 88, "values": [[4.0]]}]
    root.children.append(lamp)
    _apply_bind_pose(model)
    assert lamp.light_color == (-0.5, 0.2, 0.0)
    assert lamp.light_radius == 4.0


def test_preview_cache_key_tracks_ambient_only_and_shadow_flags() -> None:
    from src.core.lighting.preview_cache import LightmapPreviewCache

    _model, _root, mesh = _floor_model()
    cache = LightmapPreviewCache()
    settings = LightmapBakeSettings.for_kotor_engine()
    base = cache.make_key(mesh, settings, [_point()])
    assert cache.make_key(mesh, settings, [_point(ambient_only=True)]) != base
    assert cache.make_key(mesh, settings, [_point(casts_shadows=True)]) != base


# -- outdoor solver ---------------------------------------------------------------------------------------------------


def test_outdoor_encode_supports_both_composite_rules() -> None:
    from src.core.lighting import lightmap_outdoor_solver as outdoor

    texels = outdoor.PageTexels(page=0, xy=np.asarray([[0, 0], [1, 0]]), pos=np.zeros((2, 3)), nrm=np.tile([0.0, 0.0, 1.0], (2, 1)), face=np.zeros(2, int))
    comps = outdoor.OutdoorComponents(texels=[texels])
    E = [np.asarray([[0.25, 0.5, 1.0], [0.8, 0.1, 0.0]])]

    def stored(page, x, y):  # texel (x, y) -> row res - 1 - y, column x
        return page[1 - y, x, :3]

    legacy = outdoor.encode_kotor_pages(comps, E, 2, gamma=1.0, gutter=0)[0]
    assert np.array_equal(stored(legacy, 0, 0), (np.clip(E[0][0] / 2.0, 0, 1) * 255 + 0.5).astype(np.uint8))
    additive = outdoor.encode_kotor_pages(comps, E, 2, gamma=1.0, gutter=0, composite="additive", runtime_v0=0.2)[0]
    assert np.array_equal(stored(additive, 1, 0), (np.clip(E[0][1] - 0.2, 0, 1) * 255 + 0.5).astype(np.uint8))
    with pytest.raises(ValueError):
        outdoor.encode_kotor_pages(comps, E, 2, composite="multiply")


def test_outdoor_point_sun_stops_shadow_rays_at_the_sun() -> None:
    pytest.importorskip("trimesh")
    pytest.importorskip("embreex")
    from src.core.lighting import lightmap_outdoor_solver as outdoor

    # A 10 m ground square (page 0) under a roof at z = 5 (page 1).  A directional sun from straight above is
    # blocked by the roof; a point sun at z = 3 sits under the roof, so the roof must not shadow it.
    V = np.asarray(
        [[-5, -5, 0], [5, -5, 0], [5, 5, 0], [-5, 5, 0], [-6, -6, 5], [6, -6, 5], [6, 6, 5], [-6, 6, 5]], dtype=float
    )
    F = np.asarray([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6]])
    N = np.asarray([[0, 0, 1]] * 4 + [[0, 0, -1]] * 4, dtype=float)
    UV = np.asarray([[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]] * 2, dtype=float)
    face_page = np.asarray([0, 0, 1, 1])
    texels = outdoor.rasterize_pages(V, F, N, UV, face_page, 8, pages=[0])
    settings = outdoor.OutdoorBakeSettings(sun_samples=1, sky_samples=2, face_sky_samples=2)

    directional = outdoor.OutdoorLightingModel(sun_direction=(0.0, 0.0, -1.0), sun_angular_radius_deg=0.0)
    assert np.allclose(outdoor.solve_components(V, F, texels, directional, settings).sun[0], 0.0)

    sun = (1.0, -2.0, 3.0)
    point = outdoor.OutdoorLightingModel(sun_direction=(0.0, 0.0, -1.0), sun_angular_radius_deg=0.0, sun_position=sun)
    lit = outdoor.solve_components(V, F, texels, point, settings).sun[0]
    origin = texels[0].pos + texels[0].nrm * settings.normal_offset
    to_sun = np.asarray(sun)[None] - origin
    assert np.allclose(lit, to_sun[:, 2] / np.linalg.norm(to_sun, axis=1), atol=1.0e-9)

    faded = outdoor.OutdoorLightingModel(sun_direction=(0.0, 0.0, -1.0), sun_angular_radius_deg=0.0, sun_position=sun, sun_attenuation_radius=4.0)
    dist = np.linalg.norm(to_sun, axis=1)
    assert np.allclose(outdoor.solve_components(V, F, texels, faded, settings).sun[0], lit / (1.0 + 4.5 * (dist / 4.0) ** 2), atol=1.0e-9)
