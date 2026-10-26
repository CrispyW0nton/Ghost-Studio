"""Synthetic gates for Aurora animation controller semantics."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from pathlib import Path

import pytest

from src.adapters.rendering import moderngl_resources
from src.core.animation.animation_engine import (
    AnimationEngine,
    SuperModelResolver,
    _sample_controller_absolute,
    evaluate_aurora_animation_pose,
    mark_controller_times_sorted_for_sampling,
)
from src.core.geometry.lightsaber import (
    is_lightsaber_blade_node,
    is_lightsaber_model,
    lightsaber_blade_preview_quad,
    lightsaber_blade_color_choices,
    lightsaber_blade_emissive_rgb,
    lightsaber_blade_procedural_rgba8,
    lightsaber_blade_texture_cache_key,
    set_lightsaber_blade_color_override,
    should_use_procedural_lightsaber_blade_texture,
    synthetic_lightsaber_blade_uvs,
)
from src.core.geometry.model_data import Animation, GameVersion, KotorModel, ModelNode, NodeFlags
from src.core.retargeting.aurora_animation_writer import (
    AuroraAnimationInjectionRequest,
    AuroraAnimationWriter,
)
from src.core.validation.animation_block_validator import (
    AnimationBlockValidationError,
    validate_animation_block_against_model,
)
from src.gui.rendering.gpu_renderer import _build_vbo_data
from src.gui.rendering.mesh_render_data import _node_uv_array


def _quat_axis(axis: str, degrees: float) -> tuple[float, float, float, float]:
    radians = math.radians(degrees)
    s = math.sin(radians / 2.0)
    c = math.cos(radians / 2.0)
    if axis.upper() == "X":
        return (s, 0.0, 0.0, c)
    if axis.upper() == "Y":
        return (0.0, s, 0.0, c)
    if axis.upper() == "Z":
        return (0.0, 0.0, s, c)
    raise ValueError(axis)


def _quat_neg(quat: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return tuple(-value for value in quat)  # type: ignore[return-value]


def _quat_dot(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _assert_quat_equivalent(
    actual: tuple[float, float, float, float],
    expected: tuple[float, float, float, float],
    *,
    tolerance: float = 1e-6,
) -> None:
    # q and -q represent the same rotation, so compare absolute dot.
    assert abs(_quat_dot(actual, expected)) == pytest.approx(1.0, abs=tolerance)


def _animation_node(
    name: str,
    *,
    position_values: list[list[float]] | None = None,
    orientation_values: list[list[float]] | None = None,
    times: list[float] | None = None,
) -> ModelNode:
    key_times = times or [0.0]
    controllers = []
    if position_values is not None:
        controllers.append(
            {
                "type": 8,
                "name": "position",
                "columns": 3,
                "times": key_times,
                "values": position_values,
            }
        )
    if orientation_values is not None:
        controllers.append(
            {
                "type": 20,
                "name": "orientation",
                "columns": 4,
                "times": key_times,
                "values": orientation_values,
            }
        )
    return ModelNode(name=name, controllers=controllers)


def _two_node_model(
    *,
    child_position: tuple[float, float, float] = (1.0, 0.0, 0.0),
    child_rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
) -> KotorModel:
    root = ModelNode(name="root")
    child = ModelNode(name="child", position=child_position, rotation=child_rotation, parent=root)
    root.children.append(child)
    return KotorModel(name="synthetic", root_node=root, animations=[])


@pytest.fixture(autouse=True)
def _clear_supermodel_resolver():
    SuperModelResolver.clear_cache()
    SuperModelResolver.configure(None)
    yield
    SuperModelResolver.clear_cache()
    SuperModelResolver.configure(None)


def test_parent_rotation_moves_child_by_fk() -> None:
    model = _two_node_model()
    animation = Animation(
        name="pause1",
        length=1.0,
        nodes=[
            _animation_node(
                "root",
                orientation_values=[list(_quat_axis("Z", 90.0))],
            )
        ],
    )

    pose = evaluate_aurora_animation_pose(model, animation, 0.0)

    assert pose.local_transforms_by_node["child"].position == pytest.approx((1.0, 0.0, 0.0))
    assert pose.world_transforms_by_node["child"].position == pytest.approx((0.0, 1.0, 0.0), abs=1e-6)


def test_orientation_controller_is_absolute_local_not_delta() -> None:
    model = _two_node_model(child_rotation=_quat_axis("X", 30.0))
    animation = Animation(
        name="pause1",
        length=1.0,
        nodes=[_animation_node("child", orientation_values=[[0.0, 0.0, 0.0, 1.0]])],
    )

    pose = evaluate_aurora_animation_pose(model, animation, 0.0)

    _assert_quat_equivalent(pose.local_transforms_by_node["child"].rotation, (0.0, 0.0, 0.0, 1.0))


def test_position_controller_is_rest_local_delta() -> None:
    model = _two_node_model(child_position=(1.0, 0.0, 0.0))
    animation = Animation(
        name="pause1",
        length=1.0,
        nodes=[_animation_node("child", position_values=[[2.0, 0.0, 0.0]])],
    )

    pose = evaluate_aurora_animation_pose(model, animation, 0.0)

    assert pose.local_transforms_by_node["child"].position == pytest.approx((3.0, 0.0, 0.0))


def test_c_kath_root_position_controller_is_absolute_local() -> None:
    root = ModelNode(name="c_kath")
    kath_root = ModelNode(
        name="c_kathroot",
        position=(0.00186379, -0.37697601, 0.81954002),
        parent=root,
    )
    root.children.append(kath_root)
    authored_position = [0.14225900, -0.44386700, 0.81820202]
    animation = Animation(
        name="cdodgeg",
        length=1.5,
        nodes=[
            _animation_node(
                "c_kathroot",
                position_values=[authored_position, list(kath_root.position)],
                times=[0.0, 1.5],
            )
        ],
    )
    model = KotorModel(name="c_kath", root_node=root, animations=[animation])

    oracle_pose = evaluate_aurora_animation_pose(model, animation, 0.0)
    engine = AnimationEngine(model)
    assert engine.play("cdodgeg", loop=False, blend=False)
    viewport_pose = engine.evaluate(0.0)

    assert oracle_pose.local_transforms_by_node["c_kathroot"].position == pytest.approx(
        authored_position
    )
    # Ghost Studio strips the clip's first-frame root baseline so starting an
    # animation does not move the scene actor.  The format-level oracle still
    # exposes the authored absolute-local row above.
    assert viewport_pose.nodes["c_kathroot"].position == pytest.approx(kath_root.position)

    locomotion_position = [0.00186378, -0.11574100, 0.83328003]
    replacement_animation = Animation(
        name="cwalk",
        length=1.5,
        nodes=[
            _animation_node(
                "c_kathroot",
                position_values=[
                    locomotion_position,
                    [locomotion_position[0], locomotion_position[1], locomotion_position[2] + 0.1],
                ],
                times=[0.0, 1.5],
            )
        ],
    )
    replacement_model = KotorModel(
        name="c_kath",
        root_node=root,
        animations=[replacement_animation],
    )
    replacement_pose = evaluate_aurora_animation_pose(
        replacement_model,
        replacement_animation,
        0.0,
    )
    assert replacement_pose.local_transforms_by_node["c_kathroot"].position == pytest.approx(
        locomotion_position
    )

    replacement_engine = AnimationEngine(replacement_model)
    assert replacement_engine.play("cwalk", loop=False, blend=False)
    replacement_viewport_pose = replacement_engine.evaluate(0.0)
    assert replacement_viewport_pose.nodes["c_kathroot"].position == pytest.approx(
        kath_root.position
    )
    replacement_engine.seek(1.5)
    replacement_motion_pose = replacement_engine.evaluate(1.5)
    assert replacement_motion_pose.nodes["c_kathroot"].position == pytest.approx(
        (kath_root.position[0], kath_root.position[1], kath_root.position[2] + 0.1)
    )

    scene_root = ModelNode(name="Untitled Scene")
    wrapped_kath = ModelNode(name="c_kath", parent=scene_root)
    wrapped_kath_root = ModelNode(
        name="c_kathroot",
        position=kath_root.position,
        parent=wrapped_kath,
    )
    scene_root.children = [wrapped_kath]
    wrapped_kath.children = [wrapped_kath_root]
    wrapped_model = KotorModel(
        name="Untitled Scene",
        root_node=scene_root,
        animations=[replacement_animation],
    )
    wrapped_engine = AnimationEngine(wrapped_model)
    assert wrapped_engine.play("cwalk", loop=False, blend=False)
    wrapped_pose = wrapped_engine.evaluate(0.0)
    assert wrapped_pose.nodes["c_kathroot"].position == pytest.approx(kath_root.position)

    k2_animation = Animation(
        name="cdodgeg",
        length=1.5,
        nodes=[
            _animation_node(
                "c_kathroot",
                position_values=[authored_position, list(kath_root.position)],
                times=[0.0, 1.5],
            )
        ],
    )
    k2_model = KotorModel(
        name="c_kath",
        game_version=GameVersion.K2,
        root_node=root,
        animations=[k2_animation],
    )
    k2_control_pose = evaluate_aurora_animation_pose(k2_model, k2_animation, 0.0)
    assert k2_control_pose.local_transforms_by_node["c_kathroot"].position == pytest.approx(
        tuple(kath_root.position[index] + authored_position[index] for index in range(3))
    )


def _sample_position_controller(
    controller: dict[str, object],
    time_seconds: float,
) -> list[float] | None:
    return _sample_controller_absolute(
        [controller],
        8,
        "position",
        time_seconds,
        clamp=True,
    )


def test_marked_sorted_controller_uses_zero_copy_sampling() -> None:
    class IterationGuardTimes(list[float]):
        forbid_iteration = False

        def __iter__(self):
            if self.forbid_iteration:
                raise AssertionError("marked dense times were iterated during sampling")
            return super().__iter__()

    times = IterationGuardTimes([0.0, 0.5, 1.0])
    controller = {
        "type": 8,
        "name": "position",
        "times": times,
        "values": [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
    }

    assert mark_controller_times_sorted_for_sampling(controller)
    times.forbid_iteration = True
    assert _sample_position_controller(controller, 0.25) == pytest.approx([0.5, 1.0, 1.5])


def test_unmarked_integer_times_keep_float_normalization_semantics() -> None:
    controller = {
        "type": 8,
        "name": "position",
        "times": [2**53 - 1, 2**53, 2**53 + 1],
        "values": [[0.0], [1.0], [2.0]],
    }

    assert not mark_controller_times_sorted_for_sampling(controller)
    assert _sample_position_controller(controller, float(2**53)) == pytest.approx([2.0])


def test_unmarked_unsorted_tuple_rows_keep_stable_sort_semantics() -> None:
    controller = {
        "type": 8,
        "name": "position",
        "times": (1.0, 0.0, 0.0, 2.0),
        "values": ((10.0,), (1.0,), (2.0,), (20.0,)),
    }

    assert _sample_position_controller(controller, 0.0) == pytest.approx([1.0])


def test_unmarked_length_mismatch_still_truncates_rows_like_zip() -> None:
    controller = {
        "type": 8,
        "name": "position",
        "times": [0.0, 1.0, 2.0],
        "values": [[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]],
    }

    assert _sample_position_controller(controller, 2.0) == pytest.approx([2.0, 4.0, 6.0])


def test_bas_attachment_duplicate_cannot_replace_primary_body_bind_node() -> None:
    """A detachable head DAG must not supply the body's rest-local delta base."""

    body_root = ModelNode(name="body")
    body_rootdummy = ModelNode(
        name="rootdummy",
        position=(0.0, 0.0, 1.12557),
        parent=body_root,
    )
    head_socket = ModelNode(name="headhook", parent=body_rootdummy)
    attachment_root = ModelNode(name="head_attachment", parent=head_socket)
    attachment_root._gr_bas_attachment_layer = True
    attachment_root._gr_bas_attachment_root = True
    attachment_rootdummy = ModelNode(
        name="rootdummy",
        position=(0.0, 0.0, 0.0),
        parent=attachment_root,
    )
    attachment_rootdummy._gr_bas_attachment_layer = True
    body_root.children = [body_rootdummy]
    body_rootdummy.children = [head_socket]
    head_socket.children = [attachment_root]
    attachment_root.children = [attachment_rootdummy]
    animation = Animation(
        name="pause1",
        length=1.0,
        nodes=[
            _animation_node(
                "rootdummy",
                position_values=[[0.025, 0.001, -0.00665]],
            )
        ],
    )
    model = KotorModel(name="body_with_detachable_head", root_node=body_root, animations=[animation])

    engine = AnimationEngine(model)
    assert engine.play("pause1", loop=True, blend=False)
    pose = engine.evaluate(0.0)

    assert engine._base_nodes["rootdummy"] is body_rootdummy
    assert pose.nodes["rootdummy"].position == pytest.approx(
        (0.025, 0.001, 1.11892),
        abs=1.0e-6,
    )


def test_unkeyed_components_fall_back_to_rest() -> None:
    model = _two_node_model(child_position=(1.0, 2.0, 3.0), child_rotation=_quat_axis("Y", 45.0))
    animation = Animation(
        name="pause1",
        length=1.0,
        nodes=[
            _animation_node("child", orientation_values=[list(_quat_axis("Z", 10.0))]),
            _animation_node("root", position_values=[[4.0, 5.0, 6.0]]),
        ],
    )

    pose = evaluate_aurora_animation_pose(model, animation, 0.0)

    assert pose.local_transforms_by_node["child"].position == pytest.approx((1.0, 2.0, 3.0))
    _assert_quat_equivalent(pose.local_transforms_by_node["child"].rotation, _quat_axis("Z", 10.0))
    _assert_quat_equivalent(pose.local_transforms_by_node["root"].rotation, (0.0, 0.0, 0.0, 1.0))


def test_scale_controller_allows_zero_for_lightsaber_ignition() -> None:
    root = ModelNode(name="root")
    blade = ModelNode(name="plane365", parent=root, texture="w_lsabresilv01")
    root.children.append(blade)
    animation = Animation(
        name="powerup",
        length=0.8,
        nodes=[
            ModelNode(
                name="plane365",
                controllers=[
                    {
                        "type": 36,
                        "name": "scale",
                        "columns": 1,
                        "times": [0.0, 0.8],
                        "values": [[0.0], [1.0]],
                    }
                ],
            )
        ],
    )
    model = KotorModel(name="w_shortsbr_009", root_node=root, animations=[animation])
    engine = AnimationEngine(model)

    assert engine.play("powerup", loop=False, blend=False)
    assert engine.evaluate(0.0).nodes["plane365"].scale == pytest.approx(0.0)
    assert engine.evaluate(0.4).nodes["plane365"].scale == pytest.approx(0.5, abs=1e-6)
    assert engine.evaluate(0.8).nodes["plane365"].scale == pytest.approx(1.0)


def test_lightsaber_blade_material_detection_uses_blade_texture_not_hilt_name() -> None:
    hilt = ModelNode(name="LghtSbr09", texture="w_shortsbr_001")
    blade = ModelNode(name="plane365", texture="w_lsabresilv01")

    assert is_lightsaber_blade_node(hilt) is False
    assert is_lightsaber_blade_node(blade) is True
    assert max(lightsaber_blade_emissive_rgb(blade)) > 1.0


def test_lightsaber_blade_color_prefers_model_variant_over_neutral_mask() -> None:
    hilt = ModelNode(name="LghtSbr09", texture="w_shortsbr_001")
    blade = ModelNode(name="plane365", texture="w_lsabresilv01", parent=hilt)
    hilt.children.append(blade)

    r, g, b = lightsaber_blade_emissive_rgb(blade)

    assert r > b
    assert g > b
    assert r >= g


def test_lightsaber_yellow_variant_uses_procedural_texture_over_neutral_mask() -> None:
    hilt = ModelNode(name="LghtSbr09", texture="w_shortsbr_001")
    blade = ModelNode(name="plane365", texture="w_lsabresilv01", parent=hilt)
    hilt.children.append(blade)

    assert should_use_procedural_lightsaber_blade_texture(blade, texture_missing=False) is True


@pytest.mark.parametrize(
    ("texture_name", "dominant"),
    [
        ("w_lsabreblue01", "blue"),
        ("w_lsabrered01", "red"),
        ("w_lsabregren01", "green"),
        ("w_lsabreyelo01", "yellow"),
        ("w_lsabrepurp01", "violet"),
        ("w_lsabregold01", "red"),
        ("w_lsabreturq01", "blue"),
        ("w_lsabredgrn01", "green"),
        ("w_lsabresilv01", "blue"),
    ],
)
def test_stock_lightsaber_blade_textures_use_procedural_game_colour(texture_name: str, dominant: str) -> None:
    blade = ModelNode(name="plane001", texture=texture_name)

    assert should_use_procedural_lightsaber_blade_texture(blade, texture_missing=False) is True

    r, g, b = lightsaber_blade_emissive_rgb(blade)
    if dominant == "red":
        assert r > g
        assert r > b
    elif dominant == "green":
        assert g > r
        assert g > b
    elif dominant == "blue":
        assert b > r
        assert b > g
    elif dominant == "violet":
        assert b > r
        assert r > g
    elif dominant == "yellow":
        assert r > b
        assert g > b


def test_lightsaber_palette_exposes_game_blade_colours() -> None:
    color_ids = {color.id for color in lightsaber_blade_color_choices()}

    assert {
        "blue",
        "green",
        "red",
        "yellow",
        "violet",
        "viridian",
        "cyan",
        "orange",
        "bronze",
        "silver",
    } <= color_ids


def test_lightsaber_color_override_is_preview_only_and_forces_procedural_blade() -> None:
    hilt = ModelNode(name="LghtSbr09", texture="w_shortsbr_001")
    blade = ModelNode(name="plane365", texture="w_lsabresilv01", parent=hilt)
    hilt.children.append(blade)
    model = KotorModel(name="w_shortsbr_009", classification="lightsaber", root_node=hilt)

    assert is_lightsaber_model(model) is True
    before_key = lightsaber_blade_texture_cache_key(blade)
    normalized = set_lightsaber_blade_color_override(model, "blue")
    r, g, b = lightsaber_blade_emissive_rgb(blade)

    assert normalized == "blue"
    assert b > r
    assert lightsaber_blade_texture_cache_key(blade) != before_key
    assert should_use_procedural_lightsaber_blade_texture(blade, texture_missing=False) is True
    assert blade.texture == "w_lsabresilv01"


def test_lightsaber_color_override_can_be_cleared() -> None:
    hilt = ModelNode(name="LghtSbr09", texture="w_shortsbr_001")
    blade = ModelNode(name="plane365", texture="w_lsabresilv01", parent=hilt)
    hilt.children.append(blade)
    model = KotorModel(name="w_shortsbr_009", classification="lightsaber", root_node=hilt)

    set_lightsaber_blade_color_override(model, "violet")
    set_lightsaber_blade_color_override(model, None)
    r, g, b = lightsaber_blade_emissive_rgb(blade)

    assert r > b
    assert g > b


def test_lightsaber_blade_procedural_texture_has_core_and_transparent_edges() -> None:
    hilt = ModelNode(name="LghtSbr09", texture="w_shortsbr_001")
    blade = ModelNode(name="plane365", texture="w_lsabresilv01", parent=hilt)
    hilt.children.append(blade)

    width, height, rgba = lightsaber_blade_procedural_rgba8(blade, width=32, height=64)

    center_idx = ((height // 2) * width + (width // 2)) * 4
    edge_idx = ((height // 2) * width) * 4
    center = tuple(rgba[center_idx:center_idx + 4])
    edge = tuple(rgba[edge_idx:edge_idx + 4])
    assert center[0] > edge[0]
    assert center[1] > edge[1]
    assert center[3] > edge[3]
    assert edge[3] < 12


def test_lightsaber_blade_synthetic_uvs_cover_width_and_length() -> None:
    verts = [
        (-0.000177, 0.0, -0.043247),
        (-0.000177, 0.0, 0.809409),
        (0.075842, 0.0, -0.043247),
        (0.075493, 0.0, 0.780075),
    ]

    uvs = synthetic_lightsaber_blade_uvs(verts)

    assert min(u for u, _v in uvs) == pytest.approx(0.0)
    assert max(u for u, _v in uvs) == pytest.approx(1.0)
    assert min(v for _u, v in uvs) == pytest.approx(0.0)
    assert max(v for _u, v in uvs) == pytest.approx(1.0)


def test_lightsaber_blade_helpers_ignore_nonfinite_vertices() -> None:
    verts = [
        (math.nan, 0.0, 0.0),
        (math.inf, 0.0, 1.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.1, 0.0, 0.0),
    ]

    uvs = synthetic_lightsaber_blade_uvs(verts)
    quad = lightsaber_blade_preview_quad(verts)

    assert len(uvs) == 3
    assert all(math.isfinite(value) for uv in uvs for value in uv)
    assert quad is not None
    quad_verts, quad_uvs, _quad_faces, _normal = quad
    assert all(math.isfinite(value) for vertex in quad_verts for value in vertex)
    assert all(math.isfinite(value) for uv in quad_uvs for value in uv)


def test_lightsaber_blade_helpers_reject_all_nonfinite_vertices() -> None:
    verts = [(math.nan, 0.0, 0.0), (math.inf, 1.0, 0.0), ("bad", 0.0, 0.0)]

    assert synthetic_lightsaber_blade_uvs(verts) == []
    assert lightsaber_blade_preview_quad(verts) is None


def test_mesh_render_data_pads_synthetic_blade_uvs_after_bad_vertices() -> None:
    blade = ModelNode(
        name="plane365",
        texture="w_lsabresilv01",
        vertices=[
            (math.nan, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.1, 0.0, 0.0),
            (math.inf, 0.0, 0.0),
        ],
        uvs=[],
    )

    uv_arr = _node_uv_array(blade, "uvs", 5)

    assert uv_arr.shape == (5, 2)
    assert all(math.isfinite(float(value)) for row in uv_arr.tolist() for value in row)


def test_lightsaber_blade_preview_quad_fills_sparse_no_uv_variant() -> None:
    verts = [
        (0.0, 0.0, -0.043247),
        (0.0, 0.0, 0.005365),
        (0.0, 0.0, 0.932362),
        (0.0, 0.0, 0.980982),
        (0.100505, 0.0, -0.043247),
        (0.100505, 0.0, 0.005365),
        (-0.102681, 0.0, 0.932362),
        (-0.102681, 0.0, 0.980982),
    ]

    quad = lightsaber_blade_preview_quad(verts, edge_inset=0.14)

    assert quad is not None
    quad_verts, quad_uvs, quad_faces, normal = quad
    assert len(quad_verts) == 4
    assert quad_faces == [(0, 1, 2), (1, 3, 2)]
    assert min(v[2] for v in quad_verts) == pytest.approx(-0.043247)
    assert max(v[2] for v in quad_verts) == pytest.approx(0.980982)
    assert min(u for u, _v in quad_uvs) == pytest.approx(0.14)
    assert max(u for u, _v in quad_uvs) == pytest.approx(0.86)
    assert normal == (0.0, 1.0, 0.0)


def test_gpu_vbo_builder_synthesizes_blade_uvs_when_mdl_omits_them() -> None:
    blade = ModelNode(
        name="plane365",
        texture="w_lsabresilv01",
        vertices=[
            (-0.000177, 0.0, -0.043247),
            (-0.000177, 0.0, 0.809409),
            (0.075842, 0.0, -0.043247),
        ],
        normals=[(0.0, 1.0, 0.0)] * 3,
        uvs=[],
        faces=[(0, 1, 2)],
    )

    vdata, _idx = _build_vbo_data(
        blade,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )

    assert vdata is not None
    assert {tuple(row[6:8]) for row in vdata.tolist()} != {(0.5, 0.5)}


def test_gpu_vbo_builder_applies_zero_scale_to_local_blade_vertices() -> None:
    blade = ModelNode(
        name="plane365",
        texture="w_lsabresilv01",
        vertices=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.1, 0.0, 1.0)],
        normals=[(0.0, 1.0, 0.0)] * 3,
        uvs=[(0.0, 0.0), (0.0, 1.0), (1.0, 1.0)],
        faces=[(0, 1, 2)],
    )

    vdata, _idx = _build_vbo_data(
        blade,
        (1.0, 2.0, 3.0),
        (0.0, 0.0, 0.0, 1.0),
        local_scale=0.0,
    )

    assert vdata is not None
    for row in vdata[:, 0:3].tolist():
        assert row == pytest.approx([1.0, 2.0, 3.0])


def test_gpu_vbo_builder_suppresses_redundant_c_kath_hair_shell_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = ModelNode(name="c_kath")
    hair = ModelNode(
        name="Hair01",
        flags=int(NodeFlags.HEADER | NodeFlags.MESH | NodeFlags.SKIN),
        parent=root,
        texture="c_khounda02",
        vertices=[(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)],
        normals=[(0.0, 0.0, 1.0)] * 3,
        uvs=[(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)],
        faces=[(0, 1, 2)],
    )
    root.children.append(hair)
    uv_digest = hashlib.sha256(
        b"".join(struct.pack("<2f", *uv) for uv in hair.uvs)
    ).hexdigest()
    monkeypatch.setattr(moderngl_resources, "_RETAIL_C_KATH_HAIR_VERTEX_COUNT", 3)
    monkeypatch.setattr(moderngl_resources, "_RETAIL_C_KATH_HAIR_FACE_COUNT", 1)
    monkeypatch.setattr(moderngl_resources, "_RETAIL_C_KATH_HAIR_UV_SHA256", uv_digest)

    vdata, indices = _build_vbo_data(
        hair,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )

    assert vdata is None
    assert indices is None

    root.name = "c_khounda"
    control_vdata, _control_indices = _build_vbo_data(
        hair,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    assert control_vdata is not None

    root.name = "c_kath"
    hair.uvs[0] = (0.25, 0.25)
    replacement_vdata, _replacement_indices = _build_vbo_data(
        hair,
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    assert replacement_vdata is not None


def test_gpu_renderer_draws_lightsaber_blade_planes_with_culling_enabled() -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    pytest.importorskip("moderngl")
    from src.gui.rendering.gpu_renderer import GpuRenderer
    from src.gui.rendering.viewport_core import ArcBallCamera

    blade = ModelNode(
        name="plane242",
        texture="w_lsabreblue01",
        vertices=[
            (-0.05, 0.0, 0.0),
            (0.05, 0.0, 0.0),
            (-0.05, 0.0, 1.0),
            (0.05, 0.0, 1.0),
        ],
        normals=[(0.0, -1.0, 0.0)] * 4,
        uvs=[],
        faces=[(0, 1, 2), (1, 3, 2)],
    )
    model = KotorModel(
        name="w_lghtsbr_001",
        classification="lightsaber",
        root_node=blade,
    )
    renderer = GpuRenderer()
    renderer.cull_faces = True
    renderer.show_grid = False

    image = renderer.render(model, ArcBallCamera(), 300, 300, {}, None, 0.0)
    if image is None:
        pytest.skip("ModernGL context is unavailable in this environment")

    pixels = np.asarray(image.convert("RGBA"))
    blue_blade_pixels = (
        (pixels[:, :, 2] > 100)
        & (pixels[:, :, 0] < 160)
        & (pixels[:, :, 1] > 30)
        & (pixels[:, :, 3] > 0)
    )
    assert int(blue_blade_pixels.sum()) > 100


def test_quaternion_hemisphere_continuity_uses_shortest_path() -> None:
    model = _two_node_model()
    q10 = _quat_axis("Z", 10.0)
    q20_flipped = _quat_neg(_quat_axis("Z", 20.0))
    animation = Animation(
        name="pause1",
        length=1.0,
        nodes=[
            _animation_node(
                "root",
                orientation_values=[list(q10), list(q20_flipped)],
                times=[0.0, 1.0],
            )
        ],
    )

    pose = evaluate_aurora_animation_pose(model, animation, 0.5)
    q = pose.local_transforms_by_node["root"].rotation
    norm = math.sqrt(sum(value * value for value in q))
    yaw_degrees = abs(math.degrees(2.0 * math.atan2(q[2], q[3])))

    assert norm == pytest.approx(1.0, abs=1e-6)
    assert yaw_degrees == pytest.approx(15.0, abs=1.0)


def test_validator_rejects_unknown_controller_node() -> None:
    model = _two_node_model()
    animation = Animation(
        name="pause1",
        length=1.0,
        nodes=[
            _animation_node(
                "UE_Mannequin_ThighTwist_01",
                orientation_values=[[0.0, 0.0, 0.0, 1.0]],
            )
        ],
    )

    report = validate_animation_block_against_model(model, animation)

    assert report.success is False
    with pytest.raises(AnimationBlockValidationError) as exc:
        report.raise_for_errors(animation.name, model.name)
    assert "UE_Mannequin_ThighTwist_01" in str(exc.value)
    assert "KOTOR animation controllers must target existing Aurora nodes" in str(exc.value)


def test_export_path_validates_structure_before_writer(monkeypatch, tmp_path: Path) -> None:
    target_model = KotorModel(name="pmbam", animations=[Animation(name="pause1", length=1.0)])
    request = _request(tmp_path, "pause1")
    writer_called = False

    class SpyWriter:
        def write_files(self, *_args, **_kwargs):
            nonlocal writer_called
            writer_called = True

    invalid_animation = Animation(
        name="UE_Idle",
        length=1.0,
        nodes=[
            _animation_node(
                "UE_Mannequin_ThighTwist_01",
                orientation_values=[[0.0, 0.0, 0.0, 1.0]],
            )
        ],
    )

    monkeypatch.setattr(AuroraAnimationWriter, "_load_model", lambda self, req: target_model)
    monkeypatch.setattr(
        AuroraAnimationWriter,
        "build_animation_from_r3a",
        lambda self, **_kwargs: invalid_animation,
    )
    monkeypatch.setattr(
        "src.core.retargeting.aurora_animation_writer.MDLBinaryWriter",
        lambda: SpyWriter(),
    )

    result = AuroraAnimationWriter().inject(request)

    assert result.success is False
    assert writer_called is False
    assert not request.output_mdl.exists()
    assert not request.output_mdl.with_suffix(".mdx").exists()
    assert not request.output_manifest.exists()
    assert "unknown controller node 'UE_Mannequin_ThighTwist_01'" in result.errors[0]


def _request(tmp_path: Path, slot: str) -> AuroraAnimationInjectionRequest:
    r3a = tmp_path / "clip.json"
    r3a.write_text(json.dumps({"frame_count": 1, "target_curves": {}}), encoding="utf-8")
    target = tmp_path / "target.mdl"
    target.write_bytes(b"minimal target")
    return AuroraAnimationInjectionRequest(
        r3a_animation_json=r3a,
        target_mdl=target,
        animation_slot=slot,
        output_mdl=tmp_path / "out" / "target.mdl",
        output_manifest=tmp_path / "out" / "manifest.json",
    )
