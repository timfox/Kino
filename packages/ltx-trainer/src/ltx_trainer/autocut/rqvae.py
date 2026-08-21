"""Residual-quantized VAE (RQ-VAE) for video/audio discretization (Eq. 1, Supp. Table 3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ltx_trainer.autocut.config import RQVAEConfig


@dataclass
class RQVAELossBreakdown:
    reconstruction: float
    commitment: float
    cosine_sim: float

    def total(self) -> float:
        return self.reconstruction + self.commitment

    def to_dict(self) -> dict[str, float]:
        return {
            "reconstruction": self.reconstruction,
            "commitment": self.commitment,
            "cosine_sim": self.cosine_sim,
            "total": self.total(),
        }


@dataclass
class QuantizedFeatures:
    """Discrete codes + continuous reconstructions."""

    codes: list[tuple[int, int]]  # (head, code_index) per residual level
    quantized: np.ndarray
    reconstructed: np.ndarray
    embedding: np.ndarray


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).reshape(-1)
    b = b.astype(np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


def cosine_reconstruction_loss(f: np.ndarray, f_hat: np.ndarray) -> float:
    """L_rec = 1 - cos(f_hat, f) per paper Eq. (1)."""
    return 1.0 - cosine_similarity(f, f_hat)


def _fit_dim(x: np.ndarray, dim: int) -> np.ndarray:
    x = x.astype(np.float64).reshape(-1)
    if x.size == dim:
        return x
    out = np.zeros(dim, dtype=np.float64)
    n = min(dim, x.size)
    out[:n] = x[:n]
    return out


def _mlp_forward(x: np.ndarray, weights: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    h = x
    for w, b in weights:
        h = np.tanh(h @ w + b)
    return h


class ResidualRQVAE:
    """Multi-head residual vector quantizer with MLP encoder/decoder."""

    def __init__(self, cfg: RQVAEConfig, *, seed: int = 0) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        d = cfg.input_dim
        hdim = cfg.encoder_mlp[0] if cfg.encoder_mlp else d
        cb_dim = cfg.codebook_dim
        # Encoder: input_dim -> hdim -> hdim
        self._enc_w = [
            (rng.normal(scale=0.02, size=(d, hdim)), np.zeros(hdim)),
            (rng.normal(scale=0.02, size=(hdim, hdim)), np.zeros(hdim)),
        ]
        # Decoder: hdim -> hdim -> input_dim
        self._dec_w = [
            (rng.normal(scale=0.02, size=(hdim, hdim)), np.zeros(hdim)),
            (rng.normal(scale=0.02, size=(hdim, d)), np.zeros(d)),
        ]
        # Codebooks: (heads, codebook_size, codebook_dim)
        self._codebooks = rng.normal(
            scale=0.05,
            size=(cfg.quant_heads, cfg.codebook_size, cb_dim),
        ).astype(np.float64)
        self._proj_in = rng.normal(scale=0.02, size=(hdim, cb_dim))
        self._proj_out = rng.normal(scale=0.02, size=(cb_dim, hdim))

    def encode(self, features: np.ndarray) -> QuantizedFeatures:
        f = _fit_dim(features, self.cfg.input_dim)
        latent = _mlp_forward(f, self._enc_w)
        z = latent @ self._proj_in

        residual = z.copy()
        recon_z = np.zeros_like(z)
        codes: list[tuple[int, int]] = []
        scale = 1.0 / max(self.cfg.quant_heads, 1)

        for h in range(self.cfg.quant_heads):
            dists = np.linalg.norm(self._codebooks[h] - residual.reshape(1, -1), axis=1)
            code = int(np.argmin(dists))
            entry = self._codebooks[h, code]
            recon_z += entry * scale
            residual = residual - entry * scale
            codes.append((h, code))

        recon_z += residual * 0.25
        reconstructed = _mlp_forward(recon_z @ self._proj_out, self._dec_w)
        return QuantizedFeatures(
            codes=codes,
            quantized=recon_z @ self._proj_out,
            reconstructed=reconstructed,
            embedding=f,
        )

    def decode_codes(self, codes: Sequence[tuple[int, int]]) -> np.ndarray:
        """Map discrete token indices back to continuous embedding (retrieval query)."""
        cfg = self.cfg
        z = np.zeros(cfg.codebook_dim, dtype=np.float64)
        scale = 1.0 / max(cfg.quant_heads, 1)
        for h, code in codes:
            hh = int(h) % cfg.quant_heads
            cc = int(code) % cfg.codebook_size
            z += self._codebooks[hh, cc] * scale
        return _mlp_forward(z @ self._proj_out, self._dec_w)

    def forward(self, features: np.ndarray) -> RQVAELossBreakdown:
        q = self.encode(features)
        cos = cosine_similarity(q.embedding, q.reconstructed)
        rec = cosine_reconstruction_loss(q.embedding, q.reconstructed)
        return RQVAELossBreakdown(reconstruction=rec, commitment=0.0, cosine_sim=cos)


def quantize_residual(
    features: np.ndarray,
    cfg: RQVAEConfig,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Backward-compatible quantize API."""
    model = ResidualRQVAE(cfg, seed=seed)
    q = model.encode(features)
    return q.reconstructed, q.codes


def rqvae_forward(
    features: np.ndarray,
    cfg: RQVAEConfig,
    *,
    seed: int = 0,
) -> RQVAELossBreakdown:
    return ResidualRQVAE(cfg, seed=seed).forward(features)
