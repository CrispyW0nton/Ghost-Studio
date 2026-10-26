"""Binary dangling-mesh rest-position regression tests."""

from __future__ import annotations

import struct

import pytest

from src.core.geometry.model_data import (
    GameVersion,
    KotorModel,
    ModelNode,
    NodeFlags,
)
from src.core.mdl.mdl_writer import MDLBinaryWriter
from src.core.secondary_motion.profile_io import verify_raw_dangly_payload


BASE = 12
K2_TRIMESH_HEADER_SIZE = 340
DANGLY_HEADER_SIZE = 28


def _dangly_model(
    constraints: list[float] | None = None,
) -> KotorModel:
    root = ModelNode(name="dangly_test")
    hair = ModelNode(
        name="hair",
        flags=int(NodeFlags.HEADER | NodeFlags.MESH | NodeFlags.DANGLY),
        parent=root,
    )
    hair.vertices = [
        (-0.1, 0.0, 0.2),
        (0.1, 0.0, 0.2),
        (0.0, 0.0, -0.2),
    ]
    hair.normals = [(0.0, -1.0, 0.0)] * 3
    hair.uvs = [(0.0, 1.0), (1.0, 1.0), (0.5, 0.0)]
    hair.faces = [(0, 1, 2)]
    hair.face_uvs = [(0, 1, 2)]
    hair.dangly_constraints = (
        [1.0, 0.5, 0.0] if constraints is None else list(constraints)
    )
    root.children = [hair]
    return KotorModel(
        name="dangly_test",
        root_node=root,
        game_version=GameVersion.K2,
    )


def _first_child_relative_offset(mdl: bytes) -> int:
    root_rel = struct.unpack_from("<I", mdl, BASE + 40)[0]
    root_abs = BASE + root_rel
    child_array_rel = struct.unpack_from("<I", mdl, root_abs + 44)[0]
    return struct.unpack_from("<I", mdl, BASE + child_array_rel)[0]


def test_writer_emits_retail_dangly_constraint_and_rest_arrays() -> None:
    model = _dangly_model()
    mdl, _mdx = MDLBinaryWriter().write(model)

    node_abs = BASE + _first_child_relative_offset(mdl)
    header_abs = node_abs + 80 + K2_TRIMESH_HEADER_SIZE
    constraint_rel, count, count2 = struct.unpack_from("<III", mdl, header_abs)
    rest_rel = struct.unpack_from("<I", mdl, header_abs + 24)[0]

    assert count == count2 == 3
    assert constraint_rel > 0
    assert rest_rel > constraint_rel
    assert struct.unpack_from("<fff", mdl, BASE + constraint_rel) == pytest.approx(
        (255.0, 127.5, 0.0)
    )
    written_rest = [
        struct.unpack_from("<fff", mdl, BASE + rest_rel + index * 12)
        for index in range(count)
    ]
    for written, expected in zip(
        written_rest,
        model.root_node.children[0].vertices,
        strict=True,
    ):
        assert written == pytest.approx(expected)
    assert rest_rel + BASE >= constraint_rel + BASE + count * 4
    assert header_abs + DANGLY_HEADER_SIZE <= constraint_rel + BASE

    raw_verification = verify_raw_dangly_payload(
        mdl,
        header_abs=header_abs,
        vertex_count=len(model.root_node.children[0].vertices),
    )
    assert raw_verification.valid
    assert raw_verification.constraints == pytest.approx((1.0, 0.5, 0.0))
    assert raw_verification.anchor_indices == (2,)
    for written, expected in zip(
        raw_verification.rest_positions,
        model.root_node.children[0].vertices,
        strict=True,
    ):
        assert written == pytest.approx(expected)


def test_writer_rejects_dangly_constraint_vertex_count_mismatch() -> None:
    model = _dangly_model([1.0, 0.5])

    with pytest.raises(ValueError, match="one constraint per vertex"):
        MDLBinaryWriter().write(model)
