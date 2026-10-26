"""Measurable facial range-of-motion and LIP timeline gates."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from src.core.characters.facial_rig_qa import (
    audit_audio_lip_sync,
    audit_lip_timeline,
    score_facial_range,
)


def _healthy_shape_positions() -> dict[int, np.ndarray]:
    neutral = np.asarray(
        [
            [-0.02, 0.10, 0.07],
            [0.02, 0.10, 0.07],
            [-0.02, 0.10, 0.06],
            [0.02, 0.10, 0.06],
        ],
        dtype=float,
    )
    shapes = {index: neutral.copy() for index in range(16)}
    for index in range(1, 16):
        shapes[index][2:, 2] -= 0.002 + index * 0.0002
    # The canonical AH/OH poses must clearly open, while MPB closes.
    shapes[3][2:, 2] -= 0.010
    shapes[4][2:, 2] -= 0.008
    shapes[11] = neutral.copy()
    shapes[11][:2, 2] -= 0.004
    shapes[11][2:, 2] += 0.004
    return shapes


def test_range_gate_accepts_visible_open_and_closed_mouth_shapes() -> None:
    report = score_facial_range(
        model_name="p_xariah",
        shape_positions=_healthy_shape_positions(),
        upper_weights=np.asarray([1.0, 1.0, 0.0, 0.0]),
        lower_weights=np.asarray([0.0, 0.0, 1.0, 1.0]),
        mouth_weights=np.ones(4),
    )

    assert report.ok is True
    assert report.samples[3].aperture_delta > 0.01
    assert report.samples[4].aperture_delta > 0.008
    assert report.samples[11].aperture_delta < 0.0
    assert report.active_shape_count >= 12


def test_range_gate_rejects_a_controller_only_false_positive() -> None:
    neutral = _healthy_shape_positions()[0]
    report = score_facial_range(
        model_name="frozen_head",
        shape_positions={index: neutral.copy() for index in range(16)},
        upper_weights=np.asarray([1.0, 1.0, 0.0, 0.0]),
        lower_weights=np.asarray([0.0, 0.0, 1.0, 1.0]),
        mouth_weights=np.ones(4),
    )

    assert report.ok is False
    assert "frozen_facial_skin" in report.failures
    assert "ah_aperture_too_small" in report.failures


def test_range_gate_allows_only_rounding_scale_aperture_tolerance() -> None:
    shapes = _healthy_shape_positions()
    neutral = shapes[0]
    mouth_scale = float(
        np.linalg.norm(neutral.max(axis=0) - neutral.min(axis=0))
    )
    aperture_floor = mouth_scale * 0.05
    shapes[3] = neutral.copy()
    shapes[3][2:, 2] -= aperture_floor * 0.999

    near = score_facial_range(
        model_name="near_threshold_head",
        shape_positions=shapes,
        upper_weights=np.asarray([1.0, 1.0, 0.0, 0.0]),
        lower_weights=np.asarray([0.0, 0.0, 1.0, 1.0]),
        mouth_weights=np.ones(4),
    )

    assert near.ok is True
    assert "ah_aperture_near_floor" in near.warnings

    shapes[3] = neutral.copy()
    shapes[3][2:, 2] -= aperture_floor * 0.99
    below = score_facial_range(
        model_name="below_threshold_head",
        shape_positions=shapes,
        upper_weights=np.asarray([1.0, 1.0, 0.0, 0.0]),
        lower_weights=np.asarray([0.0, 0.0, 1.0, 1.0]),
        mouth_weights=np.ones(4),
    )

    assert below.ok is False
    assert "ah_aperture_too_small" in below.failures


def test_lip_timeline_reports_actual_shape_coverage_and_silence() -> None:
    lip = SimpleNamespace(
        duration=1.0,
        keyframes=[
            SimpleNamespace(time=0.0, shape=0),
            SimpleNamespace(time=0.1, shape=11),
            SimpleNamespace(time=0.2, shape=3),
            SimpleNamespace(time=0.5, shape=4),
            SimpleNamespace(time=0.8, shape=0),
        ],
    )

    report = audit_lip_timeline(lip, name="xv_test")

    assert report.ok is True
    assert report.unique_shape_count == 4
    assert report.shape_counts[3] == 1
    assert report.neutral_duration == 0.3
    assert report.active_duration_fraction == 0.7


def test_lip_timeline_rejects_a_neutral_only_file() -> None:
    lip = SimpleNamespace(
        duration=2.0,
        keyframes=[
            SimpleNamespace(time=0.0, shape=0),
            SimpleNamespace(time=2.0, shape=0),
        ],
    )

    report = audit_lip_timeline(lip, name="silent")

    assert report.ok is False
    assert "no_active_visemes" in report.failures


def test_audio_lip_sync_rejects_truncated_retail_voice() -> None:
    report = audit_audio_lip_sync(
        lip_duration=7.384,
        audio_duration=2.027755,
    )

    assert report.ok is False
    assert report.duration_delta < -5.0


def test_audio_lip_sync_accepts_container_rounding() -> None:
    report = audit_audio_lip_sync(
        lip_duration=7.384,
        audio_duration=7.383946,
    )

    assert report.ok is True
