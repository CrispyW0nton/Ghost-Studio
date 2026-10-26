"""Explicit Unreal world-centimeter to KOTOR meter conversion for level bundles.

Matrices use row-major storage and column vectors. This is not UE FMatrix layout.
The target basis reflects Y, so determinant-aware winding is required.
"""
from __future__ import annotations
import numpy as np

UE_TO_KOTOR = np.diag([0.01, -0.01, 0.01, 1.0])

def _array(value, columns: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid {name}") from exc
    if result.ndim != 2 or result.shape[1] != columns or not len(result) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be nonempty finite Nx{columns} data.")
    return result


def _faces(value, vertex_count: int) -> np.ndarray:
    result = _array(value, 3, "triangle indices")
    if (result != np.floor(result)).any() or result.min() < 0 or result.max() >= vertex_count:
        raise ValueError("Triangle index is nonintegral or outside the vertex array.")
    return result.astype(np.int64)


def _matrix(value) -> np.ndarray:
    result = _array(value, 4, "world matrix")
    if result.shape != (4, 4) or not np.allclose(result[3], [0, 0, 0, 1], atol=1e-12, rtol=0):
        raise ValueError("World matrix must be affine 4x4.")
    if abs(np.linalg.det(result[:3, :3])) < 1e-10:
        raise ValueError("Singular component transform.")
    return UE_TO_KOTOR @ result


def _geometry(mesh: dict, matrix: np.ndarray):
    vertices = _array(mesh.get("vertices"), 3, "vertices")
    if len(vertices) > 65535:
        raise ValueError("Surface exceeds the MDL u16 vertex limit; split during conditioning.")
    faces = _faces(mesh.get("triangles"), len(vertices))
    vertices = (np.c_[vertices, np.ones(len(vertices))] @ matrix.T)[:, :3]
    if np.linalg.det(matrix[:3, :3]) < 0:
        faces = faces[:, [0, 2, 1]]
    cross = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    if (np.linalg.norm(cross, axis=1) <= 1e-10).any():
        raise ValueError("Degenerate triangle after coordinate conversion.")
    return vertices, faces


def transform_normals(normals: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Object -> world normals use inverse transpose under nonuniform scale."""
    result = normals @ np.linalg.inv(matrix[:3, :3])
    lengths = np.linalg.norm(result, axis=1)
    if (lengths < 1e-12).any():
        raise ValueError("Zero-length normal.")
    return result / lengths[:, None]


def has_one_edge_connected_component(faces: np.ndarray) -> bool:
    """Connectivity for an explicitly welded indexed triangle surface."""
    parents = list(range(len(faces)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    edges = {}
    for index, (a, b, c) in enumerate(faces):
        for left, right in ((a, b), (b, c), (c, a)):
            edge = tuple(sorted((int(left), int(right))))
            if edge in edges:
                parents[root(index)] = root(edges[edge])
            else:
                edges[edge] = index
    return bool(parents) and len({root(i) for i in range(len(faces))}) == 1


