"""Surface-faithful skin-weight transfer for modular KOTOR heads.

The generic Character Builder currently offers nearest-donor-vertex weights.
Custom heads need a stronger contract: find the exact nearest donor triangle,
interpolate its three native skin rows with closest-point barycentric weights,
retain the donor palette indices, cap to Odyssey's four influences, and use a
named rigid fallback only when the workflow explicitly allows it.

This module owns only numeric transfer and row validation.  It does not clone
models, mutate a scene, interpret Qt selections, or write MDL/MDX files.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


Vec3 = tuple[float, float, float]
Face = tuple[int, int, int]


class HeadSkinTransferError(ValueError):
    """Raised when geometry or weight inputs cannot produce safe skin rows."""


@dataclass(frozen=True, slots=True)
class HeadSkinInfluence:
    palette_slot: int
    weight: float

    def __post_init__(self) -> None:
        slot = int(self.palette_slot)
        weight = float(self.weight)
        if slot < 0:
            raise HeadSkinTransferError("Palette slots must be non-negative")
        if not math.isfinite(weight) or weight <= 0.0:
            raise HeadSkinTransferError(
                "Skin influence weights must be finite and positive"
            )
        object.__setattr__(self, "palette_slot", slot)
        object.__setattr__(self, "weight", weight)

    def to_dict(self) -> dict[str, Any]:
        return {
            "palette_slot": self.palette_slot,
            "weight": self.weight,
        }


@dataclass(frozen=True, slots=True)
class HeadSkinWeightRow:
    influences: tuple[HeadSkinInfluence, ...]

    def __post_init__(self) -> None:
        if not self.influences:
            raise HeadSkinTransferError("Every target vertex requires a skin row")
        if len(self.influences) > 4:
            raise HeadSkinTransferError(
                "Odyssey skin rows support at most four influences"
            )
        slots = [row.palette_slot for row in self.influences]
        if len(slots) != len(set(slots)):
            raise HeadSkinTransferError(
                "A skin row cannot repeat a palette slot"
            )
        total = sum(row.weight for row in self.influences)
        if not math.isfinite(total) or abs(total - 1.0) > 1.0e-6:
            raise HeadSkinTransferError(
                f"Skin row weights must sum to 1.0; received {total}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "influences": [row.to_dict() for row in self.influences]
        }


@dataclass(frozen=True, slots=True)
class HeadSurfaceWeightSample:
    target_vertex_index: int
    mode: str
    donor_triangle_index: int | None
    barycentric: tuple[float, float, float] | None
    distance: float | None


@dataclass(frozen=True, slots=True)
class HeadSkinTransferReport:
    target_vertex_count: int
    surface_transfer_count: int
    explicit_rigid_count: int
    distance_fallback_count: int
    neck_floor_adjustment_count: int
    maximum_surface_distance: float
    mean_surface_distance: float
    max_surface_distance_observed: float
    max_influences_observed: int
    palette_slots_used: tuple[int, ...]
    zero_weight_vertex_count: int
    weight_rows_sha256: str

    @property
    def accepted(self) -> bool:
        return (
            self.target_vertex_count > 0
            and self.zero_weight_vertex_count == 0
            and self.max_influences_observed <= 4
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ghostrigger.head_skin_transfer",
            "version": 1,
            "accepted": self.accepted,
            "target_vertex_count": self.target_vertex_count,
            "surface_transfer_count": self.surface_transfer_count,
            "explicit_rigid_count": self.explicit_rigid_count,
            "distance_fallback_count": self.distance_fallback_count,
            "neck_floor_adjustment_count": self.neck_floor_adjustment_count,
            "maximum_surface_distance": self.maximum_surface_distance,
            "mean_surface_distance": self.mean_surface_distance,
            "max_surface_distance_observed": self.max_surface_distance_observed,
            "max_influences_observed": self.max_influences_observed,
            "palette_slots_used": list(self.palette_slots_used),
            "zero_weight_vertex_count": self.zero_weight_vertex_count,
            "weight_rows_sha256": self.weight_rows_sha256,
        }


@dataclass(frozen=True, slots=True)
class HeadSkinTransferResult:
    rows: tuple[HeadSkinWeightRow, ...]
    samples: tuple[HeadSurfaceWeightSample, ...]
    report: HeadSkinTransferReport


def transfer_head_skin_weights(
    *,
    donor_vertices: Sequence[Sequence[float]],
    donor_faces: Sequence[Sequence[int]],
    donor_weight_rows: Sequence[Any],
    target_vertices: Sequence[Sequence[float]],
    palette_size: int,
    rigid_fallback_slot: int,
    rigid_target_indices: Iterable[int] = (),
    maximum_surface_distance: float,
    allow_distance_fallback: bool = True,
    max_influences: int = 4,
    neck_target_indices: Iterable[int] = (),
    neck_palette_slot: int | None = None,
    minimum_neck_weight: float = 0.0,
) -> HeadSkinTransferResult:
    """Transfer donor triangle weights onto target vertices."""

    palette_count = int(palette_size)
    # Native KOTOR heads are not uniformly limited to 16 slots: K1's retail
    # rep_soldier_f head uses 17, and the MDX head skin tables reserve 32.
    if palette_count <= 0 or palette_count > 32:
        raise HeadSkinTransferError(
            "KOTOR head skin palettes must contain 1-32 slots"
        )
    fallback_slot = _validate_slot(rigid_fallback_slot, palette_count)
    influence_limit = max(1, min(4, int(max_influences)))
    distance_limit = float(maximum_surface_distance)
    if not math.isfinite(distance_limit) or distance_limit < 0.0:
        raise HeadSkinTransferError(
            "maximum_surface_distance must be finite and non-negative"
        )
    donor_points = _points(donor_vertices, label="donor_vertices")
    target_points = _points(target_vertices, label="target_vertices")
    if len(target_points) == 0:
        raise HeadSkinTransferError("Target head geometry contains no vertices")
    faces = _triangles(donor_faces, len(donor_points))
    if len(faces) == 0:
        raise HeadSkinTransferError("Donor head skin contains no triangles")
    donor_rows = tuple(
        normalize_head_skin_row(
            row,
            palette_size=palette_count,
            max_influences=influence_limit,
        )
        for row in donor_weight_rows
    )
    if len(donor_rows) != len(donor_points):
        raise HeadSkinTransferError(
            "Donor skin row count must match donor vertex count"
        )

    rigid = _index_set(
        rigid_target_indices,
        len(target_points),
        label="rigid_target_indices",
    )
    neck = _index_set(
        neck_target_indices,
        len(target_points),
        label="neck_target_indices",
    )
    neck_slot = (
        None
        if neck_palette_slot is None
        else _validate_slot(neck_palette_slot, palette_count)
    )
    neck_floor = float(minimum_neck_weight)
    if not math.isfinite(neck_floor) or not 0.0 <= neck_floor < 1.0:
        raise HeadSkinTransferError(
            "minimum_neck_weight must be finite in the range [0, 1)"
        )
    if neck and (neck_slot is None or neck_floor <= 0.0):
        raise HeadSkinTransferError(
            "Neck vertices require a palette slot and positive minimum weight"
        )

    triangle_points = donor_points[faces]
    triangle_minimum = triangle_points.min(axis=1)
    triangle_maximum = triangle_points.max(axis=1)
    _reject_degenerate_triangles(triangle_points)

    rows: list[HeadSkinWeightRow] = []
    samples: list[HeadSurfaceWeightSample] = []
    observed_distances: list[float] = []
    surface_count = explicit_rigid_count = distance_fallback_count = 0
    neck_adjustments = 0
    for target_index, point in enumerate(target_points):
        if target_index in rigid:
            row = HeadSkinWeightRow(
                (HeadSkinInfluence(fallback_slot, 1.0),)
            )
            sample = HeadSurfaceWeightSample(
                target_vertex_index=target_index,
                mode="explicit_rigid",
                donor_triangle_index=None,
                barycentric=None,
                distance=None,
            )
            explicit_rigid_count += 1
        else:
            triangle_index, barycentric, distance = _nearest_triangle(
                point,
                triangle_points,
                triangle_minimum,
                triangle_maximum,
            )
            observed_distances.append(distance)
            if distance > distance_limit:
                if not allow_distance_fallback:
                    raise HeadSkinTransferError(
                        f"Target vertex {target_index} is {distance:.8g} "
                        "from the donor surface, beyond the configured "
                        f"{distance_limit:.8g} limit"
                    )
                row = HeadSkinWeightRow(
                    (HeadSkinInfluence(fallback_slot, 1.0),)
                )
                mode = "distance_rigid_fallback"
                distance_fallback_count += 1
            else:
                face = faces[triangle_index]
                row = _interpolate_rows(
                    (
                        donor_rows[int(face[0])],
                        donor_rows[int(face[1])],
                        donor_rows[int(face[2])],
                    ),
                    barycentric,
                    palette_size=palette_count,
                    max_influences=influence_limit,
                )
                mode = "nearest_triangle_barycentric"
                surface_count += 1
            sample = HeadSurfaceWeightSample(
                target_vertex_index=target_index,
                mode=mode,
                donor_triangle_index=triangle_index,
                barycentric=barycentric,
                distance=distance,
            )
        if target_index in neck:
            adjusted = ensure_head_skin_influence_floor(
                row,
                palette_slot=int(neck_slot),
                minimum_weight=neck_floor,
                palette_size=palette_count,
                max_influences=influence_limit,
            )
            if adjusted != row:
                neck_adjustments += 1
            row = adjusted
        rows.append(row)
        samples.append(sample)

    result_rows = tuple(rows)
    used_slots = sorted(
        {
            influence.palette_slot
            for row in result_rows
            for influence in row.influences
        }
    )
    report = HeadSkinTransferReport(
        target_vertex_count=len(result_rows),
        surface_transfer_count=surface_count,
        explicit_rigid_count=explicit_rigid_count,
        distance_fallback_count=distance_fallback_count,
        neck_floor_adjustment_count=neck_adjustments,
        maximum_surface_distance=distance_limit,
        mean_surface_distance=(
            float(np.mean(observed_distances))
            if observed_distances
            else 0.0
        ),
        max_surface_distance_observed=max(observed_distances, default=0.0),
        max_influences_observed=max(
            (len(row.influences) for row in result_rows),
            default=0,
        ),
        palette_slots_used=tuple(used_slots),
        zero_weight_vertex_count=sum(
            not row.influences for row in result_rows
        ),
        weight_rows_sha256=head_skin_rows_sha256(result_rows),
    )
    return HeadSkinTransferResult(
        rows=result_rows,
        samples=tuple(samples),
        report=report,
    )


def normalize_head_skin_row(
    influences: Any,
    *,
    palette_size: int,
    max_influences: int = 4,
) -> HeadSkinWeightRow:
    """Coerce, merge, cap, sort, and normalize one palette-indexed row."""

    palette_count = int(palette_size)
    influence_limit = max(1, min(4, int(max_influences)))
    if isinstance(influences, HeadSkinWeightRow):
        source = influences.influences
    elif isinstance(influences, Mapping):
        source = tuple(influences.items())
    else:
        source = (
            getattr(influences, "influences", influences) or ()
        )
    merged: dict[int, float] = {}
    for raw in source:
        if isinstance(raw, HeadSkinInfluence):
            slot, weight = raw.palette_slot, raw.weight
        elif isinstance(raw, Mapping):
            slot = raw.get("palette_slot", raw.get("bone_index", -1))
            weight = raw.get("weight", 0.0)
        elif isinstance(raw, (tuple, list)) and len(raw) >= 2:
            slot, weight = raw[0], raw[1]
        else:
            slot = getattr(raw, "bone_index", -1)
            weight = getattr(raw, "weight", 0.0)
        slot = _validate_slot(slot, palette_count)
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0.0:
            raise HeadSkinTransferError(
                "Skin influence weights must be finite and non-negative"
            )
        if weight > 0.0:
            merged[slot] = merged.get(slot, 0.0) + weight
    ordered = sorted(
        merged.items(),
        key=lambda row: (-row[1], row[0]),
    )[:influence_limit]
    total = sum(weight for _slot, weight in ordered)
    if not math.isfinite(total) or total <= 1.0e-12:
        raise HeadSkinTransferError(
            "Skin row must contain at least one positive influence"
        )
    return HeadSkinWeightRow(
        tuple(
            HeadSkinInfluence(slot, weight / total)
            for slot, weight in ordered
        )
    )


def ensure_head_skin_influence_floor(
    row: HeadSkinWeightRow,
    *,
    palette_slot: int,
    minimum_weight: float,
    palette_size: int,
    max_influences: int = 4,
) -> HeadSkinWeightRow:
    """Guarantee one named palette slot while preserving a normalized row."""

    slot = _validate_slot(palette_slot, int(palette_size))
    floor = float(minimum_weight)
    if not math.isfinite(floor) or not 0.0 < floor < 1.0:
        raise HeadSkinTransferError(
            "minimum_weight must be finite in the range (0, 1)"
        )
    weights = {
        influence.palette_slot: influence.weight
        for influence in row.influences
    }
    if weights.get(slot, 0.0) >= floor:
        return row
    other = {
        index: weight
        for index, weight in weights.items()
        if index != slot and weight > 0.0
    }
    limit = max(1, min(4, int(max_influences)))
    other = dict(
        sorted(
            other.items(),
            key=lambda item: (-item[1], item[0]),
        )[: max(0, limit - 1)]
    )
    other_total = sum(other.values())
    if other_total <= 1.0e-12:
        return HeadSkinWeightRow(
            (HeadSkinInfluence(slot, 1.0),)
        )
    scaled = {
        index: (1.0 - floor) * weight / other_total
        for index, weight in other.items()
    }
    scaled[slot] = floor
    return normalize_head_skin_row(
        scaled,
        palette_size=palette_size,
        max_influences=limit,
    )


def head_skin_rows_sha256(
    rows: Sequence[HeadSkinWeightRow],
) -> str:
    payload = [
        [
            [influence.palette_slot, influence.weight]
            for influence in row.influences
        ]
        for row in rows
    ]
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _interpolate_rows(
    rows: tuple[HeadSkinWeightRow, HeadSkinWeightRow, HeadSkinWeightRow],
    barycentric: tuple[float, float, float],
    *,
    palette_size: int,
    max_influences: int,
) -> HeadSkinWeightRow:
    merged: dict[int, float] = {}
    for coefficient, row in zip(barycentric, rows):
        for influence in row.influences:
            merged[influence.palette_slot] = (
                merged.get(influence.palette_slot, 0.0)
                + float(coefficient) * influence.weight
            )
    return normalize_head_skin_row(
        merged,
        palette_size=palette_size,
        max_influences=max_influences,
    )


def _nearest_triangle(
    point: np.ndarray,
    triangles: np.ndarray,
    minimum: np.ndarray,
    maximum: np.ndarray,
) -> tuple[int, tuple[float, float, float], float]:
    below = np.maximum(minimum - point, 0.0)
    above = np.maximum(point - maximum, 0.0)
    delta = below + above
    lower_bounds = np.sum(delta * delta, axis=1)
    order = np.argsort(lower_bounds, kind="stable")
    best_index = -1
    best_distance_squared = float("inf")
    best_barycentric = (1.0, 0.0, 0.0)
    for raw_index in order:
        index = int(raw_index)
        if float(lower_bounds[index]) > best_distance_squared + 1.0e-15:
            break
        closest, barycentric = _closest_point_on_triangle(
            point,
            triangles[index, 0],
            triangles[index, 1],
            triangles[index, 2],
        )
        distance_squared = float(np.dot(point - closest, point - closest))
        if (
            distance_squared < best_distance_squared - 1.0e-15
            or (
                abs(distance_squared - best_distance_squared) <= 1.0e-15
                and (best_index < 0 or index < best_index)
            )
        ):
            best_index = index
            best_distance_squared = distance_squared
            best_barycentric = barycentric
    if best_index < 0:
        raise HeadSkinTransferError("Could not find a donor surface triangle")
    return best_index, best_barycentric, math.sqrt(best_distance_squared)


def _closest_point_on_triangle(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Return the Ericson closest point and barycentric coordinates."""

    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a, (1.0, 0.0, 0.0)

    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b, (0.0, 1.0, 0.0)

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, (1.0 - v, v, 0.0)

    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c, (0.0, 0.0, 1.0)

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, (1.0 - w, 0.0, w)

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), (0.0, 1.0 - w, w)

    denominator = 1.0 / (va + vb + vc)
    v = vb * denominator
    w = vc * denominator
    u = 1.0 - v - w
    return a + ab * v + ac * w, (u, v, w)


def _points(
    values: Sequence[Sequence[float]],
    *,
    label: str,
) -> np.ndarray:
    try:
        rows = np.asarray(values, dtype=np.float64)
    except Exception as exc:
        raise HeadSkinTransferError(f"{label} must contain numeric points") from exc
    if rows.ndim != 2 or rows.shape[1] != 3:
        raise HeadSkinTransferError(f"{label} must have shape (N, 3)")
    if not np.isfinite(rows).all():
        raise HeadSkinTransferError(f"{label} contains non-finite values")
    return rows


def _triangles(
    values: Sequence[Sequence[int]],
    vertex_count: int,
) -> np.ndarray:
    rows: list[Face] = []
    for face_index, raw in enumerate(values):
        face = tuple(int(value) for value in raw)
        if len(face) != 3:
            raise HeadSkinTransferError(
                f"Donor face {face_index} is not a triangle"
            )
        if any(index < 0 or index >= vertex_count for index in face):
            raise HeadSkinTransferError(
                f"Donor face {face_index} has an out-of-range vertex index"
            )
        rows.append(face)
    return np.asarray(rows, dtype=np.int64).reshape((-1, 3))


def _reject_degenerate_triangles(triangles: np.ndarray) -> None:
    cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    degenerate = np.where(np.sum(cross * cross, axis=1) <= 1.0e-24)[0]
    if len(degenerate):
        raise HeadSkinTransferError(
            f"Donor surface contains degenerate triangle {int(degenerate[0])}"
        )


def _validate_slot(value: Any, palette_size: int) -> int:
    try:
        slot = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HeadSkinTransferError("Palette slot must be an integer") from exc
    if slot < 0 or slot >= int(palette_size):
        raise HeadSkinTransferError(
            f"Palette slot {slot} is outside 0..{int(palette_size) - 1}"
        )
    return slot


def _index_set(
    values: Iterable[int],
    count: int,
    *,
    label: str,
) -> set[int]:
    result = {int(value) for value in values}
    invalid = sorted(index for index in result if index < 0 or index >= count)
    if invalid:
        raise HeadSkinTransferError(
            f"{label} contains out-of-range index {invalid[0]}"
        )
    return result


__all__ = [
    "HeadSkinInfluence",
    "HeadSkinTransferError",
    "HeadSkinTransferReport",
    "HeadSkinTransferResult",
    "HeadSkinWeightRow",
    "HeadSurfaceWeightSample",
    "ensure_head_skin_influence_floor",
    "head_skin_rows_sha256",
    "normalize_head_skin_row",
    "transfer_head_skin_weights",
]
