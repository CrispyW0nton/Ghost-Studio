"""Source-preserving edits to an existing Ghost Studio MDL/MDX triangle mesh."""

from dataclasses import dataclass
import struct

import numpy as np
from pykotor.resource.formats.mdl.io_mdl import _Face, _NodeHeader
from src.core.mdl.ghostrigger_mdl_reader import GhostRiggerMDLBinaryReader

MDL_BASE = 12
UNSET = 0xFFFFFFFF


@dataclass(frozen=True)
class MeshChannels:
    """Verified layout for one base-geometry node, with animations skipped."""

    header: object
    header_start: int
    mdl_position: int
    mdx_position: int
    mdx_normal: int
    stride: int
    count: int
    face_start: int
    face_indices: np.ndarray
    area_offset: int

    def positions(self, mdx):
        return self._vectors(mdx, self.mdx_position)

    def normals(self, mdx):
        return self._vectors(mdx, self.mdx_normal)

    def _vectors(self, mdx, offset):
        return np.ndarray((self.count, 3), dtype='<f4', buffer=mdx,
                          offset=offset, strides=(self.stride, 4)).astype(float)


def existing_mesh_channels(mdl, mdx, node_name):
    """Resolve channels using Ghost Studio's binary reader and verify extents."""
    reader = GhostRiggerMDLBinaryReader(mdl, source_ext=mdx, fast_load=True)
    model = reader.load()
    parsed, = [node for node in model.all_nodes()
               if node.mesh is not None and node.name.casefold() == node_name.casefold()]
    (offset, node), = [(offset, node) for offset, node in reader._gr_bin_nodes.items()
                       if node.trimesh is not None and node.header.node_id == parsed.node_id]
    h = node.trimesh
    stride, count = h.mdx_data_size, h.vertex_count
    pos, norm = h.mdx_vertex_offset, h.mdx_normal_offset
    start = MDL_BASE + offset + _NodeHeader.SIZE
    fallback = MDL_BASE + h.vertices_offset
    valid = (stride >= 24 and count > 0 and abs(pos - norm) >= 12
             and max(pos, norm) + 12 <= stride
             and h.mdx_data_offset + stride * count <= len(mdx)
             and h.vertices_offset not in (0, UNSET) and fallback + count * 12 <= len(mdl)
             and h.mdx_tangent_offset == UNSET)
    if not valid:
        raise ValueError(f'{node_name} requires equal-width MDL/MDX position and normal channels without tangent data')
    faces = np.asarray([(f.vertex1, f.vertex2, f.vertex3) for f in h.faces], dtype=int)
    face_start = MDL_BASE + h.offset_to_faces
    if (len(faces) != h.faces_count or len(faces) == 0 or faces.min() < 0
            or faces.max() >= count or face_start + _Face.SIZE * len(faces) > len(mdl)):
        raise ValueError(f'{node_name} has an incomplete triangle table')
    result = MeshChannels(h, start, fallback, h.mdx_data_offset + pos, h.mdx_data_offset + norm,
                          stride, count, face_start, faces, start + (324 if int(reader.game) == 2 else 316))
    if not np.array_equal(np.frombuffer(mdl, '<f4', count * 3, fallback).reshape(-1, 3),
                          result.positions(mdx)):
        raise ValueError(f'{node_name} has divergent source MDL and MDX positions')
    return result


def _face_vectors(vertices, faces):
    a, b, c = vertices[faces].transpose(1, 0, 2)
    cross = np.cross(b - a, c - a)
    return a, cross, np.linalg.norm(cross, axis=1)


def replace_mesh_channels(mdl, mdx, mesh, positions, normals, *, rim=None):
    """Patch existing channels, edited face planes and their mesh metadata."""
    positions = np.asarray(positions, dtype='<f4').astype(float)
    normals = np.asarray(normals, dtype='<f4').astype(float)
    original = mesh.positions(mdx)
    original_normals = mesh.normals(mdx)
    if (positions.shape != original.shape or normals.shape != original_normals.shape
            or not np.isfinite(positions).all() or not np.isfinite(normals).all()):
        raise ValueError('Replacement channels must have the original vertex count and finite values')
    moved = np.flatnonzero(np.any(positions != original, axis=1))
    relit = np.flatnonzero(np.any(normals != original_normals, axis=1))
    if len(moved) == 0 or len(relit) == 0:
        raise ValueError('Expected position and normal changes')
    triangles = mesh.face_indices
    changed_faces = np.flatnonzero(np.isin(triangles, moved).any(axis=1))
    a, cross, size = _face_vectors(positions, triangles)
    _, old_cross, old_size = _face_vectors(original, triangles)
    checked = changed_faces[old_size[changed_faces] > 1e-12]
    agreement = (old_cross[checked] * cross[checked]).sum(axis=1)
    rim_only = np.zeros(len(checked), dtype=bool) if rim is None else np.asarray(rim)[triangles[checked]].all(axis=1)
    inverted = agreement <= 0
    # Source strip closures can be made entirely of rim sites. Their orientation
    # follows the new boundary's curvature and must be checked by the cull gate.
    if np.any((inverted | (size[checked] <= 1e-12)) & ~rim_only):
        raise ValueError(f'Neck edit would collapse or flip {int(np.count_nonzero(inverted & ~rim_only))} off-rim triangles')
    out_mdl, out_mdx = bytearray(mdl), bytearray(mdx)
    for i in moved:
        for out, offset in ((out_mdl, mesh.mdl_position + i * 12),
                            (out_mdx, mesh.mdx_position + i * mesh.stride)):
            struct.pack_into('<3f', out, offset, *positions[i])
    for i in relit:
        struct.pack_into('<3f', out_mdx, mesh.mdx_normal + i * mesh.stride, *normals[i])
    for i in changed_faces[size[changed_faces] > 1e-12]:
        normal = cross[i] / size[i]
        struct.pack_into('<4f', out_mdl, mesh.face_start + i * _Face.SIZE, *normal, -np.dot(normal, a[i]))
    struct.pack_into('<f', out_mdl, mesh.area_offset, size.sum() / 2)

    # Expand stored culling extents only if a moved point exceeds them.
    # The source may store rounded or deliberately conservative bounds.
    lower = np.asarray(tuple(mesh.header.bounding_box_min))
    upper = np.asarray(tuple(mesh.header.bounding_box_max))
    if not np.isfinite([lower, upper]).all():
        raise ValueError('Invalid mesh bounds')
    if np.any(positions[moved].min(0) < lower) or np.any(positions[moved].max(0) > upper):
        lower = np.minimum(lower, positions[moved].min(0))
        upper = np.maximum(upper, positions[moved].max(0))
        struct.pack_into('<6f', out_mdl, mesh.header_start + 20, *lower, *upper)
        # The Ghost Studio writer stores a conservative radius around the origin.
        radius = max(mesh.header.radius, np.linalg.norm(positions[moved], axis=1).max())
        struct.pack_into('<f', out_mdl, mesh.header_start + 44, radius)
    return bytes(out_mdl), bytes(out_mdx), {'moved_vertices': len(moved),
            'relit_vertices': len(relit), 'edited_face_planes': len(changed_faces),
            'rim_only_reoriented_triangles': int(np.count_nonzero(inverted & rim_only)),
            'rim_only_degenerate_triangles': int(np.count_nonzero((size[checked] <= 1e-12) & rim_only))}
