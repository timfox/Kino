"""Partition mesh stitching (Sec. 3.2, Eq. 7–8)."""

from __future__ import annotations

import torch
from torch import Tensor


def clip_mesh_to_exterior(
    vertices: Tensor,
    faces: Tensor,
    opposing_bounds_min: Tensor,
    opposing_bounds_max: Tensor,
) -> tuple[Tensor, Tensor]:
    """M1\\2: keep vertices outside opposing bounding volume (toy axis-aligned)."""
    inside = (
        (vertices[:, 0] >= opposing_bounds_min[0])
        & (vertices[:, 0] <= opposing_bounds_max[0])
        & (vertices[:, 1] >= opposing_bounds_min[1])
        & (vertices[:, 1] <= opposing_bounds_max[1])
        & (vertices[:, 2] >= opposing_bounds_min[2])
        & (vertices[:, 2] <= opposing_bounds_max[2])
    )
    keep = ~inside
    if keep.sum() < 3:
        return vertices, faces
    idx_map = -torch.ones(vertices.shape[0], dtype=torch.long)
    idx_map[keep] = torch.arange(int(keep.sum()))
    new_v = vertices[keep]
    valid_faces = []
    for f in faces:
        if keep[f[0]] and keep[f[1]] and keep[f[2]]:
            valid_faces.append(idx_map[f].tolist())
    if not valid_faces:
        return new_v, torch.zeros(0, 3, dtype=torch.long)
    return new_v, torch.tensor(valid_faces, dtype=torch.long)


def seam_vertices_in_overlap(
    v1: Tensor,
    b1_min: Tensor,
    b1_max: Tensor,
    b2_min: Tensor,
    b2_max: Tensor,
) -> Tensor:
    """Vertices in overlap O = B1 ∩ B2."""
    in_b2 = (
        (v1[:, 0] >= b2_min[0])
        & (v1[:, 0] <= b2_max[0])
        & (v1[:, 1] >= b2_min[1])
        & (v1[:, 1] <= b2_max[1])
        & (v1[:, 2] >= b2_min[2])
        & (v1[:, 2] <= b2_max[2])
    )
    in_b1 = (
        (v1[:, 0] >= b1_min[0])
        & (v1[:, 0] <= b1_max[0])
        & (v1[:, 1] >= b1_min[1])
        & (v1[:, 1] <= b1_max[1])
        & (v1[:, 2] >= b1_min[2])
        & (v1[:, 2] <= b1_max[2])
    )
    return v1[in_b2 & in_b1]


def stitch_meshes_vertex_count(v1: Tensor, v2: Tensor, seam: Tensor) -> int:
    """Toy stitch: union vertex counts (no full Delaunay)."""
    return int(v1.shape[0] + v2.shape[0] + seam.shape[0])
