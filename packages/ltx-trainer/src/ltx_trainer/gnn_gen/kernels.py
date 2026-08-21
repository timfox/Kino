"""Graph NNGP and Graph NTK for GCN (Theorem 3; Sabanayagam et al., Du et al.)."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from ltx_trainer.gnn_gen.convolution import symmetric_normalized_adjacency


def _relu_second_moment(sigma: Tensor) -> Tensor:
    """E[relu(z_i) relu(z_j)] for z ~ N(0, Sigma), arc-cosine kernel (Cho & Saul)."""
    n = sigma.shape[0]
    diag = torch.diag(sigma).clamp(min=1e-12)
    sii = diag.unsqueeze(1).expand(n, n)
    sjj = diag.unsqueeze(0).expand(n, n)
    rho = (sigma / (sii * sjj).sqrt().clamp(min=1e-12)).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.arccos(rho)
    scale = (sii * sjj).sqrt() / (2 * math.pi)
    off = scale * (torch.sin(theta) + (math.pi - theta) * rho)
    eye = torch.eye(n, device=sigma.device, dtype=torch.bool)
    return torch.where(eye, 0.5 * diag.unsqueeze(1).expand(n, n), off)


def _relu_derivative_covariance(sigma: Tensor) -> Tensor:
    """E[relu'(z_i) relu'(z_j)] = P(z_i > 0, z_j > 0) for centred Gaussian."""
    n = sigma.shape[0]
    diag = torch.diag(sigma).clamp(min=1e-12)
    rho = (sigma / (diag.unsqueeze(1).sqrt() * diag.unsqueeze(0).sqrt().clamp(min=1e-12))).clamp(
        -1.0 + 1e-6, 1.0 - 1e-6
    )
    off = 0.25 + torch.arcsin(rho) / (2 * math.pi)
    eye = torch.eye(n, device=sigma.device, dtype=torch.bool)
    return torch.where(eye, torch.full((n, n), 0.5, device=sigma.device, dtype=sigma.dtype), off)


def graph_nngp_gcn(
    x: Tensor,
    adj: Tensor,
    *,
    n_layers: int,
    s: Tensor | None = None,
) -> Tensor:
    """Node-level Graph NNGP kernel K_nngp = E_L (Theorem 3).

    ``x``: (n, d0) node features; ``adj``: (n, n).
    """
    if s is None:
        s = symmetric_normalized_adjacency(adj)
    sigma = s @ x @ x.T @ s.T
    for _ in range(n_layers):
        e = _relu_second_moment(sigma)
        sigma = s @ e @ s.T
    return sigma


def graph_ntk_gcn(
    x: Tensor,
    adj: Tensor,
    *,
    n_layers: int,
    s: Tensor | None = None,
) -> Tensor:
    """Node-level Graph NTK (Theorem 3).

    K_ntk = sum_{k=0}^{L-1} S(...(Sigma_k ⊙ dot_E_k)...) ⊙ dot_E_{L-1} with L-1-k convolutions.
    """
    if s is None:
        s = symmetric_normalized_adjacency(adj)
    sigmas: list[Tensor] = [s @ x @ x.T @ s.T]
    dot_es: list[Tensor] = []
    for _ in range(n_layers):
        dot_es.append(_relu_derivative_covariance(sigmas[-1]))
        sigmas.append(s @ _relu_second_moment(sigmas[-1]) @ s.T)

    l = n_layers
    k_ntk = torch.zeros_like(sigmas[0])
    for k_idx in range(l):
        block = sigmas[k_idx] * dot_es[k_idx]
        for r in range(1, l - k_idx):
            block = s @ block @ s.T * dot_es[k_idx + r]
        k_ntk = k_ntk + block
    return k_ntk
