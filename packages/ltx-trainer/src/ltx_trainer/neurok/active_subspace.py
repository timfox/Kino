"""Active subspace dimension reduction (Sec. 4.3, Constantine et al.)."""

from __future__ import annotations

import torch
from torch import Tensor


def deformation_norm_proxy(z: Tensor, decoder_weight: Tensor) -> Tensor:
    """G(z) = ||δ_pred(z)||_2 as surrogate for active-subspace sensitivity."""
    delta = z @ decoder_weight.T
    return delta.norm(dim=-1)


def active_subspace_basis(
    decoder_weight: Tensor,
    *,
    reduced_dim: int,
    num_probe: int = 32,
) -> Tensor:
    """Estimate dominant directions via gradient covariance of G(z)."""
    z_probe = torch.randn(num_probe, decoder_weight.shape[1])
    grads: list[Tensor] = []
    for z in z_probe:
        z_i = z.detach().clone().requires_grad_(True)
        g = deformation_norm_proxy(z_i.unsqueeze(0), decoder_weight)
        g.backward()
        grads.append(z_i.grad.detach())
    grad_mat = torch.stack(grads, dim=0)
    cov = grad_mat.T @ grad_mat / max(num_probe - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = torch.argsort(eigvals, descending=True)
    k = min(reduced_dim, eigvecs.shape[1])
    return eigvecs[:, order[:k]].T  # (kq, k)


def project_latent(z: Tensor, basis: Tensor) -> Tensor:
    """Map z ∈ R^k to reduced coordinates q ∈ R^{kq}."""
    return z @ basis.T


def lift_latent(q: Tensor, basis: Tensor) -> Tensor:
    """Map q back to full latent (linear subspace)."""
    return q @ basis
