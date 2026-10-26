"""See-through audit for single-sided game meshes (KOTOR draws one side of every face).

Added 2026-09-28 from the DXM_101 (Dathomir) conversion. An Unreal master renders many surfaces two-sided (Tripo
materials are doubleSided by default, foliage and thin shells too), and a converter that cuts or reduces faces can
open gaps; KOTOR then shows the void or the sky through the model. The user asked for "zero culled faces all around".
This audit casts rays from every place the player's camera can be and reports two kinds of defect:

- back-face hits: the first surface a ray meets faces away from the eye. KOTOR culls it, so the eye sees through it.
  ``double_sided_fix`` appends those faces flipped, which closes every such view;
- escapes below the horizon: a ray pointing down that meets nothing - a hole to the void (no floor, no backing).
  These need geometry, not a flag; the report gives the eye and direction of each so they can be found.

Eyes inside closed geometry are rejected first (most of a probe's first hits are back faces). Headless and file-free:
arrays in, arrays out; uses trimesh's Embree intersector when available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def fibonacci_directions(count: int) -> np.ndarray:
    """``count`` unit vectors spread evenly over the sphere."""

    i = np.arange(count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * i / count
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    return np.c_[r * np.cos(phi), r * np.sin(phi), z]


def _intersector(vertices: np.ndarray, faces: np.ndarray) -> Any:
    import trimesh

    mesh = trimesh.Trimesh(vertices, faces, process=False)
    try:
        return trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    except Exception:  # pragma: no cover - Embree missing
        return trimesh.ray.ray_triangle.RayMeshIntersector(mesh)


@dataclass
class SeeThroughReport:
    eyes_used: int
    eyes_rejected_inside: int
    rays: int
    backface_hits: int
    backface_faces: np.ndarray                      # unique face indices hit from behind
    backface_counts: np.ndarray                     # rays per face in ``backface_faces``
    escaped_down: int
    escaped_down_samples: list = field(default_factory=list)   # (eye, direction) of up to 200 escapes

    def summary(self) -> dict:
        return {'eyesUsed': self.eyes_used, 'eyesRejectedInside': self.eyes_rejected_inside, 'rays': self.rays,
                'backfaceHits': self.backface_hits, 'backfaceFaces': int(len(self.backface_faces)),
                'escapedDown': self.escaped_down}


def audit_single_sided(vertices: np.ndarray, faces: np.ndarray, eyes: np.ndarray, *, rays_per_eye: int = 4096,
                       below_horizon: float = -0.08, inside_probe: int = 64, inside_ratio: float = 0.5) -> SeeThroughReport:
    """Cast ``rays_per_eye`` rays from each eye over the mesh as a one-sided renderer would draw it."""

    V = np.asarray(vertices, np.float64); F = np.asarray(faces, np.int64)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    ray = _intersector(V, F)
    probe = fibonacci_directions(inside_probe)
    dirs = fibonacci_directions(rays_per_eye)
    counts = np.zeros(len(F), np.int64)
    used = rejected = back = down = 0
    samples: list = []
    for eye in np.asarray(eyes, np.float64):
        _, pr, pt = ray.intersects_location(np.repeat(eye[None], len(probe), 0), probe, multiple_hits=False)
        if len(pt) and (np.einsum('ij,ij->i', probe[pr], fn[pt]) > 0).sum() > inside_ratio * len(probe):
            rejected += 1
            continue
        used += 1
        _, ir, it = ray.intersects_location(np.repeat(eye[None], len(dirs), 0), dirs, multiple_hits=False)
        bk = np.einsum('ij,ij->i', dirs[ir], fn[it]) > 0
        np.add.at(counts, it[bk], 1); back += int(bk.sum())
        hit = np.zeros(len(dirs), bool); hit[ir] = True
        esc = np.nonzero(~hit & (dirs[:, 2] < below_horizon))[0]
        down += len(esc)
        for j in esc[:max(0, 200 - len(samples))]:
            samples.append((eye.round(3).tolist(), dirs[j].round(3).tolist()))
    faces_hit = np.nonzero(counts)[0]
    return SeeThroughReport(used, rejected, used * len(dirs), back, faces_hit, counts[faces_hit], down, samples)


def double_sided_fix(faces: np.ndarray, face_ids: np.ndarray) -> np.ndarray:
    """``faces`` with the given faces appended again, reversed (drawn from their back as well)."""

    F = np.asarray(faces, np.int64)
    ids = np.unique(np.asarray(face_ids, np.int64))
    return np.r_[F, F[ids][:, ::-1]]


__all__ = ('SeeThroughReport', 'audit_single_sided', 'double_sided_fix', 'fibonacci_directions')
