"""Focused immutable-contract tests for selected native head donors."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.core.characters.head_donor_snapshot import (
    HeadDonorSnapshot,
    capture_head_donor_snapshot,
    compare_head_donor_contract,
    validate_head_donor_snapshot,
)
from src.core.geometry.model_data import (
    Animation,
    BoneWeight,
    GameVersion,
    KotorModel,
    ModelNode,
    NodeFlags,
    VertexSkinData,
)


def _attach(parent: ModelNode, child: ModelNode) -> None:
    parent.children.append(child)
    child.parent = parent


def _native_head_model() -> KotorModel:
    root = ModelNode(name="PFHA04", flags=int(NodeFlags.HEADER), index=0, number=0)
    neck = ModelNode(name="neck_g", flags=int(NodeFlags.HEADER), index=1, number=30)
    head_bone = ModelNode(
        name="head_g",
        flags=int(NodeFlags.HEADER),
        index=2,
        number=32,
    )
    jaw = ModelNode(
        name="f_jaw_g",
        flags=int(NodeFlags.HEADER),
        index=3,
        number=37,
    )
    skin = ModelNode(
        name="head",
        flags=int(NodeFlags.HEADER | NodeFlags.MESH | NodeFlags.SKIN),
        index=4,
        number=355,
        vertices=[
            (-0.1, 0.0, 0.0),
            (0.1, 0.0, 0.0),
            (0.0, 0.0, 0.2),
        ],
        normals=[(0.0, -1.0, 0.0)] * 3,
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)],
        faces=[(0, 1, 2)],
        texture="PFHA04",
        bone_map=["head_g", "neck_g", "f_jaw_g"],
        skin_data=[
            VertexSkinData([BoneWeight(0, 1.0)]),
            VertexSkinData([BoneWeight(0, 0.75), BoneWeight(1, 0.25)]),
            VertexSkinData([BoneWeight(2, 1.0)]),
        ],
        bb_min=(-0.1, 0.0, 0.0),
        bb_max=(0.1, 0.0, 0.2),
        radius=0.15,
    )
    skin.bone_node_indices = [2, 1, 3]
    for node in (neck, head_bone, jaw, skin):
        _attach(root, node)

    node_count = 5
    skin.qbone_list = [
        (0.0, 0.0, 0.0, 1.0)
        for _ in range(node_count)
    ]
    skin.tbone_list = [
        (0.0, 0.0, 0.0)
        for _ in range(node_count)
    ]
    model = KotorModel(
        name="PFHA04",
        supermodel="S_Female03",
        game_version=GameVersion.K2,
        model_type=4,
        root_node=root,
        bb_min=(-5.0, -5.0, -1.0),
        bb_max=(5.0, 5.0, 10.0),
        radius=7.0,
        super_root_node_name="neck_g",
        geometry_node_count=564,
    )
    model._gr_render_bounds = ((-0.1, 0.0, 0.0), (0.1, 0.0, 0.2))
    model._gr_render_radius = 0.15
    return model


def _snapshot(model: KotorModel | None = None) -> HeadDonorSnapshot:
    return capture_head_donor_snapshot(
        model or _native_head_model(),
        game="K2",
        resref="PFHA04",
        resource_view="stock_only",
        mdl_sha256="a" * 64,
        mdx_sha256="b" * 64,
        provenance={
            "stock": True,
            "effective_override": False,
            "source": "chitin:models.bif",
        },
    )


def test_native_head_snapshot_is_immutable_serializable_and_eligible() -> None:
    snapshot = _snapshot()

    assert snapshot.geometry_root_name == "PFHA04"
    assert snapshot.attachment_target_name == "neck_g"
    assert snapshot.inherited_node_declaration == 564
    assert snapshot.local_node_count == 5
    assert snapshot.mutable_payload_node_ordinals == (4,)
    assert [row.sparse_node_number for row in snapshot.nodes] == [
        0,
        30,
        32,
        37,
        355,
    ]
    assert snapshot.skins[0].bone_palette == (
        "head_g",
        "neck_g",
        "f_jaw_g",
    )
    assert snapshot.skins[0].bind_row_count == 5
    assert snapshot.retail_bb_min == (-5.0, -5.0, -1.0)
    assert snapshot.preview_bb_max == (0.1, 0.0, 0.2)
    assert validate_head_donor_snapshot(snapshot).eligible is True

    restored = HeadDonorSnapshot.from_dict(snapshot.to_dict())
    assert restored == snapshot
    assert restored.structural_sha256 == snapshot.structural_sha256


def test_secondary_tongue_skin_does_not_require_head_or_neck_influences() -> None:
    model = _native_head_model()
    root = model.root_node
    head = model.find_node("head")
    assert root is not None and head is not None
    tongue = ModelNode(
        name="tongue",
        flags=int(NodeFlags.HEADER | NodeFlags.MESH | NodeFlags.SKIN),
        index=5,
        number=356,
        vertices=[
            (-0.01, -0.05, 0.02),
            (0.01, -0.05, 0.02),
            (0.0, -0.06, 0.03),
        ],
        normals=[(0.0, -1.0, 0.0)] * 3,
        uvs=[(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)],
        faces=[(0, 1, 2)],
        texture="PFHA04",
        bone_map=["f_jaw_g"],
        skin_data=[VertexSkinData([BoneWeight(0, 1.0)]) for _ in range(3)],
    )
    tongue.bone_node_indices = [3]
    tongue.qbone_list = [(0.0, 0.0, 0.0, 1.0)] * 6
    tongue.tbone_list = [(0.0, 0.0, 0.0)] * 6
    head.qbone_list.append((0.0, 0.0, 0.0, 1.0))
    head.tbone_list.append((0.0, 0.0, 0.0))
    _attach(root, tongue)

    report = validate_head_donor_snapshot(_snapshot(model))

    assert report.eligible is True


def test_geometry_material_and_weights_can_change_without_changing_donor_dag() -> None:
    source = _native_head_model()
    snapshot = _snapshot(source)
    edited = deepcopy(source)
    skin = edited.find_node("head")
    assert skin is not None
    skin.vertices = [
        (-0.2, 0.0, 0.0),
        (0.2, 0.0, 0.0),
        (0.0, 0.0, 0.3),
        (0.0, -0.1, 0.1),
    ]
    skin.normals = [(0.0, -1.0, 0.0)] * 4
    skin.uvs = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0), (0.5, 0.5)]
    skin.faces = [(0, 1, 2), (0, 3, 1)]
    skin.skin_data = [
        VertexSkinData([BoneWeight(0, 1.0)]),
        VertexSkinData([BoneWeight(0, 0.5), BoneWeight(1, 0.5)]),
        VertexSkinData([BoneWeight(2, 1.0)]),
        VertexSkinData([BoneWeight(0, 0.9), BoneWeight(1, 0.1)]),
    ]
    skin.texture = "P_CUSTOMH"
    edited._gr_render_bounds = ((-0.2, -0.1, 0.0), (0.2, 0.0, 0.3))
    edited._gr_render_radius = 0.25

    diff = compare_head_donor_contract(snapshot, edited)

    assert diff.structurally_compatible is True
    assert diff.blocking == ()
    assert any(row.path == "meshes[4]" for row in diff.allowed_payload_changes)
    assert any(
        row.path == "skins[0].influence_sha256"
        for row in diff.allowed_payload_changes
    )
    assert any(
        row.path == "preview_bb_max"
        for row in diff.allowed_payload_changes
    )


def test_explicit_output_resref_may_rename_only_model_and_geometry_root() -> None:
    source = _native_head_model()
    snapshot = _snapshot(source)
    output = deepcopy(source)
    output.name = "P_CUSTOMH"
    assert output.root_node is not None
    output.root_node.name = "P_CUSTOMH"

    allowed = compare_head_donor_contract(
        snapshot,
        output,
        output_resref="P_CUSTOMH",
    )
    denied = compare_head_donor_contract(snapshot, output)

    assert allowed.structurally_compatible is True
    assert {
        row.path for row in allowed.allowed_payload_changes
    } >= {"model_name", "geometry_root_name", "nodes[0].name"}
    assert denied.structurally_compatible is False
    assert {
        row.path for row in denied.blocking
    } >= {"model_name", "geometry_root_name", "nodes[0].name"}


def test_local_animation_addition_requires_explicit_allow_list() -> None:
    source = _native_head_model()
    snapshot = _snapshot(source)
    edited = deepcopy(source)
    edited.animations = [Animation(name="talk")]

    denied = compare_head_donor_contract(snapshot, edited)
    allowed = compare_head_donor_contract(
        snapshot,
        edited,
        allowed_local_animation_names=("talk",),
    )

    assert "local_animation_names" in {
        row.path for row in denied.blocking
    }
    assert allowed.blocking == ()
    assert any(
        row.path == "local_animation_names"
        and row.category == "animation_payload"
        for row in allowed.allowed_payload_changes
    )


@pytest.mark.parametrize(
    ("mutate", "expected_path"),
    [
        (
            lambda model: setattr(model, "geometry_node_count", 38),
            "inherited_node_declaration",
        ),
        (
            lambda model: setattr(model, "super_root_node_name", "head_g"),
            "attachment_target_name",
        ),
        (
            lambda model: setattr(model, "bb_max", (1.0, 1.0, 1.0)),
            "retail_bb_max",
        ),
        (
            lambda model: setattr(
                model.find_node("head"),
                "qbone_list",
                [(1.0, 0.0, 0.0, 0.0)] * 5,
            ),
            "skins[0].qbone_rows",
        ),
        (
            lambda model: setattr(model.find_node("neck_g"), "number", 999),
            "nodes[1].sparse_node_number",
        ),
    ],
)
def test_critical_donor_changes_are_blocking(mutate, expected_path: str) -> None:
    source = _native_head_model()
    snapshot = _snapshot(source)
    changed = deepcopy(source)
    mutate(changed)

    diff = compare_head_donor_contract(snapshot, changed)

    assert diff.structurally_compatible is False
    assert expected_path in {row.path for row in diff.blocking}


def test_parent_change_invalidates_unchanged_dag_proof() -> None:
    source = _native_head_model()
    snapshot = _snapshot(source)
    changed = deepcopy(source)
    root = changed.root_node
    skin = changed.find_node("head")
    head_bone = changed.find_node("head_g")
    assert root is not None and skin is not None and head_bone is not None
    root.children.remove(skin)
    head_bone.children.append(skin)
    skin.parent = head_bone

    diff = compare_head_donor_contract(snapshot, changed)

    assert diff.structurally_compatible is False
    assert any(
        row.path.endswith(("parent_ordinal", "child_ordinals"))
        for row in diff.blocking
    )


def test_geometry_outside_selected_head_skin_is_not_mutable_payload() -> None:
    source = _native_head_model()
    helper = ModelNode(
        name="eyeRA",
        flags=int(NodeFlags.HEADER | NodeFlags.MESH),
        index=5,
        number=356,
        vertices=[(0.02, -0.08, 0.1)],
        normals=[(0.0, -1.0, 0.0)],
        uvs=[(0.5, 0.5)],
        texture="PFHA04EYE",
    )
    assert source.root_node is not None
    _attach(source.root_node, helper)
    skin = source.find_node("head")
    assert skin is not None
    skin.qbone_list.append((0.0, 0.0, 0.0, 1.0))
    skin.tbone_list.append((0.0, 0.0, 0.0))
    snapshot = _snapshot(source)
    changed = deepcopy(source)
    changed_helper = changed.find_node("eyeRA")
    assert changed_helper is not None
    changed_helper.vertices = [(9.0, 9.0, 9.0)]

    diff = compare_head_donor_contract(snapshot, changed)

    assert diff.structurally_compatible is False
    assert "meshes[5]" in {row.path for row in diff.blocking}


def test_eligibility_reports_palette_weight_and_attachment_failures() -> None:
    model = _native_head_model()
    skin = model.find_node("head")
    assert skin is not None
    skin.bone_map = [f"bone_{index}" for index in range(17)]
    skin.bone_node_indices = list(range(17))
    skin.skin_data[0] = VertexSkinData(
        [
            BoneWeight(0, 0.2),
            BoneWeight(1, 0.2),
            BoneWeight(2, 0.2),
            BoneWeight(3, 0.2),
            BoneWeight(4, -0.1),
        ]
    )
    model.super_root_node_name = "missing_neck"

    report = validate_head_donor_snapshot(_snapshot(model))
    checks = {row.check_id for row in report.issues}

    assert report.eligible is False
    assert "head.donor.attachment_target" in checks
    assert "head.donor.palette_limit" in checks
    assert "head.donor.influences_finite" in checks
    assert "head.donor.influences_normalized" in checks
    assert "head.donor.head_influence" in checks


def test_contract_tampering_is_detected_on_deserialization() -> None:
    payload = _snapshot().to_dict()
    payload["supermodel"] = "S_Male03"

    with pytest.raises(ValueError, match="fingerprint"):
        HeadDonorSnapshot.from_dict(payload)


def test_stock_policy_rejects_override_provenance() -> None:
    snapshot = replace(
        _snapshot(),
        provenance={"effective_override": True},
    )

    report = validate_head_donor_snapshot(snapshot)

    assert report.eligible is False
    assert "head.donor.stock_provenance" in {
        row.check_id for row in report.issues
    }
