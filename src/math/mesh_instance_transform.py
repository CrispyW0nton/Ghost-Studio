"""Bake object-space mesh attributes through ordered scene transforms."""
from __future__ import annotations

import numpy as np

from src.math.gpu_math import _matrix_from_pos_quat_np
from src.math.camera_math import euler_degrees_to_quat


def bake_mesh_instance(vertices, normals, faces, *, position, rotation, transforms=()):
    """Apply node object→model, then room→module→world transforms.

    Node rotation is XYZW; scene rotations are XYZ Euler degrees. Normals use
    the inverse transpose. Reflections reverse winding to preserve front faces.
    """
    matrix = np.asarray(_matrix_from_pos_quat_np(position, rotation), dtype=float)
    for transform in transforms:
        if transform is None:
            continue
        layer = np.asarray(_matrix_from_pos_quat_np(
            transform.position, euler_degrees_to_quat(transform.rotation)
        ), dtype=float)
        layer[:3, :3] = layer[:3, :3] @ np.diag(transform.scale)
        matrix = layer @ matrix
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
        raise ValueError("Mesh instance has a non-finite or singular transform")
    points = np.asarray(vertices, dtype=float)
    if not np.isfinite(points).all():
        raise ValueError("Mesh contains non-finite vertices")
    points = points @ matrix[:3, :3].T + matrix[:3, 3]
    vectors = np.asarray(normals, dtype=float)
    if not np.isfinite(vectors).all():
        raise ValueError("Mesh contains non-finite normals")
    if len(vectors) == len(points):
        vectors = vectors @ np.linalg.inv(matrix[:3, :3])
        lengths = np.linalg.norm(vectors, axis=1)
        vectors = vectors / np.maximum(lengths[:, None], 1e-12)
    result_faces = [tuple(face) for face in faces]
    if np.linalg.det(matrix[:3, :3]) < 0:
        result_faces = [(a, c, b) for a, b, c in result_faces]
    return points.tolist(), vectors.tolist(), result_faces
