"""Visual-audit camera contracts for translated modular-head skins."""

from __future__ import annotations

from scripts.audit_head_facial_performance import _facial_frame_bounds
from src.core.geometry.model_data import (
    BoneWeight,
    KotorModel,
    ModelNode,
    VertexSkinData,
)


def test_facial_frame_bounds_use_world_space_for_translated_head_skin() -> None:
    root = ModelNode(name="PMHA04")
    skin = ModelNode(
        name="head",
        position=(0.0, 0.0, 1.7),
        vertices=[
            (-0.04 + index * 0.004, 0.08, -1.64 + (index % 3) * 0.01)
            for index in range(20)
        ],
        bone_map=["f_um_g"],
        skin_data=[
            VertexSkinData([BoneWeight(0, 1.0)]) for _ in range(20)
        ],
    )
    root.children.append(skin)
    skin.parent = root
    model = KotorModel(name="PMHA04", root_node=root)

    minimum, maximum = _facial_frame_bounds(model)

    assert minimum[2] > -0.1
    assert maximum[2] < 0.2
