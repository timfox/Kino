"""SE(3) helpers: quaternions and 4×4 transforms (Sec. III-A)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def quat_normalize(q: Tensor) -> Tensor:
    return F.normalize(q, dim=-1, eps=1e-8)


def quat_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """Hamilton product; q layout (w, x, y, z)."""
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_to_matrix(q: Tensor) -> Tensor:
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], dim=-1),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], dim=-1),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], dim=-1),
        ],
        dim=-2,
    )


def transform_from_qt(q: Tensor, t: Tensor) -> Tensor:
    """Build [B, 4, 4] homogeneous transforms from quaternion + translation."""
    b = q.shape[0]
    r = quat_to_matrix(q)
    t4 = torch.zeros(b, 4, 4, device=q.device, dtype=q.dtype)
    t4[:, :3, :3] = r
    t4[:, :3, 3] = t
    t4[:, 3, 3] = 1.0
    return t4


def transform_points(T: Tensor, points: Tensor) -> Tensor:
    """Apply [B,4,4] to [B,N,3]."""
    ones = torch.ones(*points.shape[:-1], 1, device=points.device, dtype=points.dtype)
    homo = torch.cat([points, ones], dim=-1)
    out = torch.einsum("bij,bnj->bni", T, homo)
    return out[..., :3]


def compose_transforms(*mats: Tensor) -> Tensor:
    out = mats[0]
    for m in mats[1:]:
        out = torch.bmm(out, m)
    return out


def identity_transform(batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    eye = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(batch, -1, -1)
    return eye.clone()


def quat_geodesic_distance(q_pred: Tensor, q_gt: Tensor) -> Tensor:
    """Angular distance proxy (radians) per batch element."""
    q_pred = quat_normalize(q_pred)
    q_gt = quat_normalize(q_gt)
    inner = (q_pred * q_gt).sum(dim=-1).abs().clamp(-1.0, 1.0)
    return 2.0 * torch.acos(inner)
