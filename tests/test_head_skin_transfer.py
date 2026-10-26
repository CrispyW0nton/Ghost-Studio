from __future__ import annotations

import pytest

from src.math.head_skin_transfer import (
    HeadSkinTransferError,
    ensure_head_skin_influence_floor,
    normalize_head_skin_row,
    transfer_head_skin_weights,
)


DONOR_VERTICES = (
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
)
DONOR_FACES = ((0, 1, 2),)
DONOR_ROWS = (
    ((0, 1.0),),
    ((1, 1.0),),
    ((2, 1.0),),
)


def test_barycentric_transfer_interpolates_native_palette_rows():
    result = transfer_head_skin_weights(
        donor_vertices=DONOR_VERTICES,
        donor_faces=DONOR_FACES,
        donor_weight_rows=DONOR_ROWS,
        target_vertices=((0.25, 0.25, 0.1),),
        palette_size=4,
        rigid_fallback_slot=3,
        maximum_surface_distance=0.2,
    )

    weights = {
        row.palette_slot: row.weight
        for row in result.rows[0].influences
    }
    assert weights == pytest.approx({0: 0.5, 1: 0.25, 2: 0.25})
    assert result.samples[0].barycentric == pytest.approx(
        (0.5, 0.25, 0.25)
    )
    assert result.samples[0].distance == pytest.approx(0.1)
    assert result.report.surface_transfer_count == 1
    assert result.report.distance_fallback_count == 0
    assert result.report.accepted


def test_distance_and_explicit_rigid_fallbacks_are_distinguished():
    result = transfer_head_skin_weights(
        donor_vertices=DONOR_VERTICES,
        donor_faces=DONOR_FACES,
        donor_weight_rows=DONOR_ROWS,
        target_vertices=((0.2, 0.2, 2.0), (0.3, 0.3, 0.0)),
        palette_size=4,
        rigid_fallback_slot=3,
        rigid_target_indices=(1,),
        maximum_surface_distance=0.25,
    )

    assert result.rows[0].influences[0].palette_slot == 3
    assert result.rows[1].influences[0].palette_slot == 3
    assert result.samples[0].mode == "distance_rigid_fallback"
    assert result.samples[1].mode == "explicit_rigid"
    assert result.report.distance_fallback_count == 1
    assert result.report.explicit_rigid_count == 1


def test_distance_limit_can_be_a_blocking_gate():
    with pytest.raises(HeadSkinTransferError, match="beyond"):
        transfer_head_skin_weights(
            donor_vertices=DONOR_VERTICES,
            donor_faces=DONOR_FACES,
            donor_weight_rows=DONOR_ROWS,
            target_vertices=((0.2, 0.2, 2.0),),
            palette_size=4,
            rigid_fallback_slot=3,
            maximum_surface_distance=0.25,
            allow_distance_fallback=False,
        )


def test_neck_floor_adds_required_attachment_influence_and_caps_row():
    row = normalize_head_skin_row(
        ((0, 0.4), (1, 0.3), (2, 0.2), (3, 0.1)),
        palette_size=5,
    )
    adjusted = ensure_head_skin_influence_floor(
        row,
        palette_slot=4,
        minimum_weight=0.2,
        palette_size=5,
    )
    weights = {
        influence.palette_slot: influence.weight
        for influence in adjusted.influences
    }

    assert len(weights) == 4
    assert weights[4] == pytest.approx(0.2)
    assert 3 not in weights
    assert sum(weights.values()) == pytest.approx(1.0)


def test_transfer_applies_neck_floor_and_records_adjustments():
    result = transfer_head_skin_weights(
        donor_vertices=DONOR_VERTICES,
        donor_faces=DONOR_FACES,
        donor_weight_rows=DONOR_ROWS,
        target_vertices=((0.25, 0.25, 0.0),),
        palette_size=4,
        rigid_fallback_slot=3,
        maximum_surface_distance=0.1,
        neck_target_indices=(0,),
        neck_palette_slot=3,
        minimum_neck_weight=0.1,
    )

    weights = {
        influence.palette_slot: influence.weight
        for influence in result.rows[0].influences
    }
    assert weights[3] == pytest.approx(0.1)
    assert result.report.neck_floor_adjustment_count == 1


def test_invalid_palette_rows_or_donor_geometry_are_rejected():
    with pytest.raises(HeadSkinTransferError, match="outside"):
        normalize_head_skin_row(((4, 1.0),), palette_size=4)
    with pytest.raises(HeadSkinTransferError, match="degenerate"):
        transfer_head_skin_weights(
            donor_vertices=((0, 0, 0), (1, 0, 0), (2, 0, 0)),
            donor_faces=((0, 1, 2),),
            donor_weight_rows=DONOR_ROWS,
            target_vertices=((0, 0, 0),),
            palette_size=4,
            rigid_fallback_slot=3,
            maximum_surface_distance=1.0,
        )


def test_retail_seventeen_slot_head_palette_is_supported():
    result = transfer_head_skin_weights(
        donor_vertices=DONOR_VERTICES,
        donor_faces=DONOR_FACES,
        donor_weight_rows=DONOR_ROWS,
        target_vertices=((0.25, 0.25, 0.0),),
        palette_size=17,
        rigid_fallback_slot=16,
        maximum_surface_distance=0.1,
    )
    assert result.report.accepted

    with pytest.raises(HeadSkinTransferError, match="1-32"):
        transfer_head_skin_weights(
            donor_vertices=DONOR_VERTICES,
            donor_faces=DONOR_FACES,
            donor_weight_rows=DONOR_ROWS,
            target_vertices=((0.25, 0.25, 0.0),),
            palette_size=33,
            rigid_fallback_slot=32,
            maximum_surface_distance=0.1,
        )


def test_transfer_is_deterministic_for_equal_distance_triangle_ties():
    kwargs = dict(
        donor_vertices=(
            *DONOR_VERTICES,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        donor_faces=((0, 1, 2), (3, 4, 5)),
        donor_weight_rows=(
            *DONOR_ROWS,
            ((3, 1.0),),
            ((3, 1.0),),
            ((3, 1.0),),
        ),
        target_vertices=((0.2, 0.2, 0.0),),
        palette_size=4,
        rigid_fallback_slot=3,
        maximum_surface_distance=0.1,
    )
    first = transfer_head_skin_weights(**kwargs)
    second = transfer_head_skin_weights(**kwargs)

    assert first.samples[0].donor_triangle_index == 0
    assert first.report.weight_rows_sha256 == (
        second.report.weight_rows_sha256
    )
