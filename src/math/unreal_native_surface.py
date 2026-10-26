"""Native Unreal render-section to normalized interchange math, before world bake."""
from __future__ import annotations
import numpy as np
from .unreal_level_space import _array, _faces


def normalize_native_section(section: dict, material: str, uv_scale=(1.0, 1.0)) -> dict:
    vertices = _array(section.get("vertices"), 3, "native vertices")
    faces = _faces(section.get("triangles"), len(vertices))[:, [0, 2, 1]]
    normals = _array(section.get("normals"), 3, "native normals")
    uvs = _array(section.get("uvs"), 2, "native UVs")
    scale = np.asarray(uv_scale, dtype=float)
    if scale.shape != (2,) or not np.isfinite(scale).all():
        raise ValueError("UV scale must be two finite values.")
    if len(normals) != len(vertices) or len(uvs) != len(vertices):
        raise ValueError("Native split vertices, normals and UVs must agree.")
    uvs = uvs * scale
    uvs[:, 1] = 1 - uvs[:, 1]
    return {"vertices": vertices.tolist(), "triangles": faces.tolist(),
            "normals": normals.tolist(), "uvs": uvs.tolist(), "material": material}


def linear_rgb_to_srgb8(value) -> tuple[int, int, int]:
    rgb = np.asarray(value, dtype=float)
    if rgb.shape != (3,) or not np.isfinite(rgb).all() or (rgb < 0).any() or (rgb > 1).any():
        raise ValueError("HDR/out-of-range constants need an explicit tone-map bake.")
    encoded = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.power(rgb, 1 / 2.4) - 0.055)
    return tuple(int(round(c * 255)) for c in encoded)


def propose_section_spawn(section, world_matrix):
    """Suggest a point on an upward-facing triangle in Unreal world centimeters."""
    from .unreal_level_space import _matrix
    matrix = _matrix(world_matrix)
    vertices = _array(section["vertices"], 3, "native vertices")
    vertices = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    faces = _faces(section["triangles"], len(vertices))[:, [0, 2, 1]]
    if np.linalg.det(matrix[:3, :3]) < 0:
        faces = faces[:, [0, 2, 1]]
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    candidates = np.flatnonzero((lengths > 1e-8) & (normals[:, 2] >= lengths * 0.7))
    if not len(candidates):
        raise ValueError("This section has no upward-facing floor triangle. Choose an authored floor section.")
    index = candidates[np.argmax(lengths[candidates])]
    return triangles[index].mean(axis=0).tolist()
