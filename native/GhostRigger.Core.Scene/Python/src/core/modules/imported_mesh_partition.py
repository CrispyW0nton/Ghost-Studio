"""Split imported room surfaces that are past the engine's per-node vertex bound.

KOTOR mesh nodes index vertices with 16-bit values, and reverse engineering of swkotor2.exe found signed consumers:
a render node past 32,767 exported vertices is a crash hazard even though the file format could hold 65,535
(Dathomir TE01 technical evaluation, 2026-11-17). ``MDL_MAX_VERTICES_PER_SURFACE`` is that hard bound; this module
does the split that the validator asks for.

Splitting never cuts a triangle and never moves a vertex. A surface is divided into compact groups of its own
triangles (median cuts along the longest axis), and each group keeps copies of the vertices it uses, with every
per-vertex channel (normals, UVs, lightmap UVs) and per-face material carried over. Pieces that meet share identical
positions and normals along the seam, so the room renders as before.

Surface indices are stable: the first piece stays at the original index and extra pieces are appended at the end,
so index-based roles (``render``, ``imported_srf_<n>``) and reviewed walkmesh selections of unsplit surfaces keep
their meaning. A split surface's reviewed face selection is not remapped: split before reviewing floors.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .authored_imported_mesh import (
    MDL_MAX_VERTICES_PER_SURFACE,
    ImportedMeshRoomPrimitive,
    ImportedMeshSurface,
)

#: Export target below the hard bound: leaves room for vertices added by UV and lightmap seams.
MDL_VERTEX_TARGET_PER_SURFACE = 30000


def _groups(vertices: np.ndarray, faces: np.ndarray, max_vertices: int) -> list[tuple[np.ndarray, np.ndarray]]:
    centers = vertices[faces].mean(axis=1)
    done, stack = [], [np.arange(len(faces))]
    while stack:
        index = stack.pop()
        used = np.unique(faces[index])
        if len(used) <= max_vertices:
            done.append((index, used))
            continue
        if len(index) < 2:
            raise ValueError("A single triangle cannot exceed the vertex bound; face data is corrupt.")
        center = centers[index]
        axis = int(np.argmax(center.max(axis=0) - center.min(axis=0)))
        order = np.argsort(center[:, axis], kind="stable")
        half = len(index) // 2
        stack.append(index[order[half:]])
        stack.append(index[order[:half]])
    done.sort(key=lambda item: int(item[0].min()))      # the piece holding face 0 comes first: deterministic
    return done


def split_imported_mesh_surface(
    surface: ImportedMeshSurface, *, max_vertices: int = MDL_VERTEX_TARGET_PER_SURFACE
) -> tuple[ImportedMeshSurface, ...]:
    """Return the surface unchanged if it fits, else compact pieces of at most ``max_vertices`` vertices each."""
    if not 3 <= max_vertices <= MDL_MAX_VERTICES_PER_SURFACE:
        raise ValueError(f"max_vertices must be between 3 and {MDL_MAX_VERTICES_PER_SURFACE}.")
    count = len(surface.vertices)
    if count <= max_vertices:
        return (surface,)
    vertices = np.asarray(surface.vertices, dtype=float)
    faces = np.asarray(surface.faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.min() < 0 or faces.max() >= count:
        raise ValueError(f"Surface {surface.name} has invalid face indices; repair it before splitting.")

    def per_vertex(channel, used):
        return tuple(channel[int(i)] for i in used) if len(channel) == count else channel

    pieces = []
    for number, (index, used) in enumerate(_groups(vertices, faces, max_vertices)):
        remap = np.full(count, -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        local = remap[faces[index]]
        face_mats = (
            tuple(surface.face_mats[int(i)] for i in index) if len(surface.face_mats) == len(faces) else surface.face_mats
        )
        pieces.append(replace(
            surface,
            name=surface.name if number == 0 else f"{surface.name}_p{number:02d}",
            vertices=per_vertex(surface.vertices, used),
            faces=tuple((int(a), int(b), int(c)) for a, b, c in local),
            face_mats=face_mats,
            uvs=per_vertex(surface.uvs, used),
            normals=per_vertex(surface.normals, used),
            uvs_lm=per_vertex(surface.uvs_lm, used),
        ))
    return tuple(pieces)


def split_oversized_imported_mesh_surfaces(
    primitive: ImportedMeshRoomPrimitive, *, max_vertices: int = MDL_VERTEX_TARGET_PER_SURFACE
) -> ImportedMeshRoomPrimitive:
    """Split every surface of a room that is past ``max_vertices``. Unsplit surfaces keep their index."""
    kept, extra = [], []
    for surface in primitive.surfaces:
        pieces = split_imported_mesh_surface(surface, max_vertices=max_vertices)
        kept.append(pieces[0])
        extra.extend(pieces[1:])
    if not extra:
        return primitive
    metadata = {**primitive.metadata, "vertex_bound_split": {"max_vertices": max_vertices, "added_surfaces": len(extra)}}
    return replace(primitive, surfaces=tuple(kept) + tuple(extra), metadata=metadata)


def oversized_imported_mesh_surfaces(
    primitive: ImportedMeshRoomPrimitive, *, max_vertices: int = MDL_MAX_VERTICES_PER_SURFACE
) -> tuple[tuple[int, str, int], ...]:
    """(index, name, vertex count) of every surface past ``max_vertices``; empty when the room is safe to export."""
    return tuple(
        (index, surface.name, len(surface.vertices))
        for index, surface in enumerate(primitive.surfaces)
        if len(surface.vertices) > max_vertices
    )


__all__ = [
    "MDL_VERTEX_TARGET_PER_SURFACE",
    "oversized_imported_mesh_surfaces",
    "split_imported_mesh_surface",
    "split_oversized_imported_mesh_surfaces",
]
