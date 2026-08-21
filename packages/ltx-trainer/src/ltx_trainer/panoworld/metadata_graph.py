"""Verified metadata graph G=(V,E) (Eq. 4–6)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class EntityNode:
    """vi = (si, ai, bi, di, ci) — semantics, attrs, BFOV, depth, context."""

    semantics: Tensor  # [C] logits or one-hot
    attributes: Tensor  # [A]
    bfov: Tensor  # [4] yaw, pitch, xfov, yfov (degrees)
    depth: Tensor  # scalar observer distance
    context: Tensor  # [D] visual feature


@dataclass
class MetadataGraph:
    nodes: list[EntityNode]
    edges: Tensor  # [E, 6] Δθ, Δφ, Δd, r2d, r3d, pad

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)


def _discretized_relation(delta_yaw: float, delta_pitch: float, delta_d: float) -> tuple[int, int]:
    """Coarse spherical / 3D relation bins for supervision stub."""
    if delta_yaw < -30:
        r2d = 0
    elif delta_yaw > 30:
        r2d = 2
    else:
        r2d = 1
    if delta_d < -0.5:
        r3d = 0
    elif delta_d > 0.5:
        r3d = 2
    else:
        r3d = 1
    return r2d, r3d


def build_metadata_graph(
    semantics: Tensor,
    bfov: Tensor,
    depth: Tensor,
    context: Tensor,
    *,
    attributes: Tensor | None = None,
) -> MetadataGraph:
    """
  Build graph from batched entity tensors.

  semantics [N,C], bfov [N,4], depth [N], context [N,D].
    """
    n = semantics.shape[0]
    if attributes is None:
        attributes = torch.zeros(n, 4, device=semantics.device, dtype=semantics.dtype)
    nodes: list[EntityNode] = []
    for i in range(n):
        nodes.append(
            EntityNode(
                semantics=semantics[i],
                attributes=attributes[i],
                bfov=bfov[i],
                depth=depth[i],
                context=context[i],
            )
        )
    edge_rows: list[list[float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            dy = float(bfov[j, 0] - bfov[i, 0])
            dp = float(bfov[j, 1] - bfov[i, 1])
            dd = float(depth[j] - depth[i])
            r2d, r3d = _discretized_relation(dy, dp, dd)
            edge_rows.append([dy, dp, dd, float(r2d), float(r3d), 0.0])
    if edge_rows:
        edges = torch.tensor(edge_rows, device=semantics.device, dtype=semantics.dtype)
    else:
        edges = torch.zeros(0, 6, device=semantics.device, dtype=semantics.dtype)
    return MetadataGraph(nodes=nodes, edges=edges)
