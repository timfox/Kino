"""Quaternion helpers for rigid propagation (Eq. 4–5), wxyz convention."""

from __future__ import annotations

import torch
from torch import Tensor


def quat_normalize(q: Tensor, eps: float = 1e-8) -> Tensor:
    return q / (q.norm(dim=-1, keepdim=True).clamp(min=eps))


def quat_mul(q: Tensor, r: Tensor) -> Tensor:
    """Hamilton product ``q ⊗ r``; both (..., 4) wxyz."""
    qw, qx, qy, qz = q.unbind(-1)
    rw, rx, ry, rz = r.unbind(-1)
    w = qw * rw - qx * rx - qy * ry - qz * rz
    x = qw * rx + qx * rw + qy * rz - qz * ry
    y = qw * ry - qx * rz + qy * rw + qz * rx
    z = qw * rz + qx * ry - qy * rx + qz * rw
    return quat_normalize(torch.stack([w, x, y, z], dim=-1))


def quat_to_rotmat(q: Tensor) -> Tensor:
    """Rotation matrix (..., 3, 3) from unit quaternion wxyz."""
    q = quat_normalize(q)
    w, x, y, z = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m00 = 1 - 2 * (yy + zz)
    m01 = 2 * (xy - wz)
    m02 = 2 * (xz + wy)
    m10 = 2 * (xy + wz)
    m11 = 1 - 2 * (xx + zz)
    m12 = 2 * (yz - wx)
    m20 = 2 * (xz - wy)
    m21 = 2 * (yz + wx)
    m22 = 1 - 2 * (xx + yy)
    r0 = torch.stack([m00, m01, m02], dim=-1)
    r1 = torch.stack([m10, m11, m12], dim=-1)
    r2 = torch.stack([m20, m21, m22], dim=-1)
    return torch.stack([r0, r1, r2], dim=-2)


def quat_rotate_vector(q: Tensor, v: Tensor) -> Tensor:
    """Apply rotation ``q`` to vectors ``v``; ``v`` is (..., 3)."""
    r = quat_to_rotmat(q)
    return torch.matmul(v.unsqueeze(-2), r.transpose(-1, -2)).squeeze(-2)
