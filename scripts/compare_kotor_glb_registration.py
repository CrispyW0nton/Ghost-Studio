"""Measure rigid facial-part registration against a stock KOTOR head reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"Point shape mismatch: {source.shape} vs {target.shape}")
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    left, _singular, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1, :] *= -1.0
        rotation = right_t.T @ left.T
    translation = target_center - rotation @ source_center
    fitted = (rotation @ source.T).T + translation
    error = np.linalg.norm(fitted - target, axis=1)
    return rotation, translation, float(error.max(initial=0.0))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock", type=Path, required=True)
    parser.add_argument("--interchange", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    stock_payload = json.loads(args.stock.read_text(encoding="utf-8"))
    interchange_payload = json.loads(args.interchange.read_text(encoding="utf-8"))
    stock = {str(row["name"]).casefold(): row for row in stock_payload["nodes"]}
    interchange = {
        str(row["name"]).casefold(): row for row in interchange_payload["meshes"]
    }
    stock_head = np.asarray(stock["head"]["world_vertices"], dtype=np.float64)
    interchange_head = np.asarray(interchange["head"]["vertices_world"], dtype=np.float64)
    rotation, translation, head_fit_max = _rigid_fit(stock_head, interchange_head)

    rows = []
    for name in ("head", "eyela", "eyera", "eyellid", "eyerlid", "teethla", "teethua", "tongue"):
        source = np.asarray(stock[name]["world_vertices"], dtype=np.float64)
        target = np.asarray(interchange[name]["vertices_world"], dtype=np.float64)
        expected = (rotation @ source.T).T + translation
        error = np.linalg.norm(expected - target, axis=1)
        own_rotation, own_translation, own_fit_max = _rigid_fit(source, target)
        rows.append(
            {
                "name": name,
                "vertex_count": int(len(source)),
                "head_frame_error_mean": float(error.mean()),
                "head_frame_error_max": float(error.max(initial=0.0)),
                "own_rigid_fit_error_max": own_fit_max,
                "own_translation_delta_from_head": [
                    float(value) for value in own_translation - translation
                ],
                "own_rotation_delta_max_abs": float(np.max(np.abs(own_rotation - rotation))),
            }
        )
    payload = {
        "schema": "ghostrigger.kotor-glb-registration-comparison.v1",
        "stock": str(args.stock),
        "interchange": str(args.interchange),
        "head_fit_max": head_fit_max,
        "head_rotation": rotation.tolist(),
        "head_translation": translation.tolist(),
        "meshes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
