"""Regression coverage for KOTOR-to-glTF animation transform semantics."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest


def _position_controller(delta_x: float) -> dict[str, object]:
    return {
        "type": 8,
        "times": [0.0],
        "values": [[delta_x, 0.0, 0.0]],
    }


def _accessor_vec3(gltf, accessor_index: int) -> tuple[float, float, float]:
    accessor = gltf.accessors[accessor_index]
    view = gltf.bufferViews[accessor.bufferView]
    offset = int(view.byteOffset or 0) + int(accessor.byteOffset or 0)
    return struct.unpack_from("<fff", gltf.binary_blob(), offset)


def _accessor_mat4(gltf, accessor_index: int) -> tuple[float, ...]:
    accessor = gltf.accessors[accessor_index]
    view = gltf.bufferViews[accessor.bufferView]
    offset = int(view.byteOffset or 0) + int(accessor.byteOffset or 0)
    return struct.unpack_from("<16f", gltf.binary_blob(), offset)


def test_gltf_position_keys_add_bind_position_and_scale() -> None:
    from src.converters.mesh_converter import _gltf_absolute_translation_samples

    samples = _gltf_absolute_translation_samples(
        (10.0, 20.0, 30.0),
        [(1.0, -2.0, 0.5)],
        2.0,
    )
    assert samples[0] == pytest.approx([12.0, 16.0, 31.0])


def test_gltf_duplicate_node_animation_targets_are_unique_and_occurrence_stable(
    tmp_path: Path,
) -> None:
    pygltflib = pytest.importorskip("pygltflib")
    from src.converters.mesh_converter import GLTFExporter
    from src.core.geometry import model_data as md

    root = md.ModelNode(name="root", position=(0.0, 0.0, 0.0))
    first = md.ModelNode(name="piece", position=(10.0, 0.0, 0.0), parent=root)
    second = md.ModelNode(name="piece", position=(20.0, 0.0, 0.0), parent=root)
    root.children = [first, second]

    first_track = md.ModelNode(name="piece", controllers=[_position_controller(1.0)])
    second_track = md.ModelNode(name="piece", controllers=[_position_controller(2.0)])
    animation = md.Animation(
        name="walk",
        length=1.0,
        nodes=[first_track, second_track],
    )
    model = md.KotorModel(
        name="duplicate_parts",
        root_node=root,
        animations=[animation],
        anim_scale=1.0,
    )

    output = tmp_path / "duplicate_parts.glb"
    assert GLTFExporter()._export_pygltflib(model, str(output), True)

    gltf = pygltflib.GLTF2().load(str(output))
    assert [node.name for node in gltf.nodes] == [
        "root",
        "piece",
        "piece__dup2",
    ]
    channels = [
        channel
        for channel in gltf.animations[0].channels
        if channel.target.path == "translation"
    ]
    assert [channel.target.node for channel in channels] == [1, 2]
    translations = [
        _accessor_vec3(gltf, gltf.animations[0].samplers[channel.sampler].output)
        for channel in channels
    ]
    assert translations == pytest.approx([(11.0, 0.0, 0.0), (22.0, 0.0, 0.0)])


def test_gltf_uses_distinct_kotor_qbone_palettes_per_skin_mesh(tmp_path: Path) -> None:
    pygltflib = pytest.importorskip("pygltflib")
    from src.converters.mesh_converter import GLTFExporter
    from src.core.geometry import model_data as md

    root = md.ModelNode(name="root")
    bone = md.ModelNode(name="bone", position=(1.0, 0.0, 0.0), parent=root)
    root.children = [bone]

    def skin_mesh(name: str, inverse_bind_x: float) -> md.ModelNode:
        node = md.ModelNode(
            name=name,
            flags=int(md.NodeFlags.HEADER | md.NodeFlags.MESH | md.NodeFlags.SKIN),
            parent=root,
            vertices=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            normals=[(0.0, 0.0, 1.0)] * 3,
            uvs=[(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
            faces=[(0, 1, 2)],
            bone_map=["bone"],
            skin_data=[
                md.VertexSkinData([md.BoneWeight(0, 1.0)]) for _ in range(3)
            ],
        )
        node.bone_node_indices = [1]
        # Full DFS-indexed Odyssey arrays. qBone is raw WXYZ and the selected
        # row is consumed directly as the glTF inverse bind.
        node.qbone_list = [(1.0, 0.0, 0.0, 0.0)] * 4
        node.tbone_list = [
            (0.0, 0.0, 0.0),
            (inverse_bind_x, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        ]
        return node

    first = skin_mesh("skin_a", 5.0)
    second = skin_mesh("skin_b", 9.0)
    root.children.extend([first, second])
    model = md.KotorModel(name="per_mesh_palettes", root_node=root)

    output = tmp_path / "per_mesh_palettes.glb"
    assert GLTFExporter()._export_pygltflib(model, str(output), True)

    gltf = pygltflib.GLTF2().load(str(output))
    assert len(gltf.skins) == 2
    assert [skin.joints for skin in gltf.skins] == [[1], [1]]
    first_bind = _accessor_mat4(gltf, gltf.skins[0].inverseBindMatrices)
    second_bind = _accessor_mat4(gltf, gltf.skins[1].inverseBindMatrices)
    assert first_bind[12:15] == pytest.approx((5.0, 0.0, 0.0))
    assert second_bind[12:15] == pytest.approx((9.0, 0.0, 0.0))
    mesh_nodes = [node for node in gltf.nodes if node.name in {"skin_a", "skin_b"}]
    assert [node.skin for node in mesh_nodes] == [0, 1]
