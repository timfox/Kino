"""Optional PyTorch linear VAE for HoliTok Stage II (falls back when torch absent)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.holitok.config import HoliTokConfig
from ltx_trainer.holitok.encoder import encode as numpy_encode
from ltx_trainer.holitok.vae import kl_gaussian, stage_ii_vae_loss


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def build_torch_vae(cfg: HoliTokConfig | None = None, *, seed: int = 0):
    """Return a tiny linear VAE module or None if torch is unavailable."""
    if not _torch_available():
        return None
    import torch
    import torch.nn as nn

    cfg = cfg or HoliTokConfig()
    rng = torch.Generator().manual_seed(seed)

    class LinearVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enc = nn.Linear(cfg.encoder_hop, cfg.latent_dim, bias=True)
            self.dec = nn.Linear(cfg.latent_dim, cfg.encoder_hop, bias=True)
            nn.init.normal_(self.enc.weight, std=0.02, generator=rng)
            nn.init.normal_(self.dec.weight, std=0.02, generator=rng)

        def encode(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            mean = self.enc(frames)
            log_var = torch.log1p(frames.var(dim=-1, keepdim=True) + 1e-6).expand_as(mean)
            return mean, log_var

        def decode(self, latent: torch.Tensor) -> torch.Tensor:
            return self.dec(latent)

        def forward(self, frames: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mean, log_var = self.encode(frames)
            z = mean
            recon = self.decode(z)
            return recon, mean, log_var

    return LinearVAE()


def torch_encode_decode(wave: np.ndarray, *, cfg: HoliTokConfig | None = None, seed: int = 0) -> dict[str, Any]:
    cfg = cfg or HoliTokConfig()
    model = build_torch_vae(cfg, seed=seed)
    if model is None:
        enc = numpy_encode(wave, cfg=cfg, seed=seed)
        return {"backend": "numpy", "latent_shape": list(enc["latent"].shape)}

    import torch

    wave = np.asarray(wave, dtype=np.float64).ravel()
    hop = cfg.encoder_hop
    n = max(1, (wave.size + hop - 1) // hop)
    frames = np.zeros((n, hop), dtype=np.float32)
    for i in range(n):
        chunk = wave[i * hop : (i + 1) * hop]
        if chunk.size < hop:
            chunk = np.pad(chunk, (0, hop - chunk.size))
        frames[i] = chunk

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(frames)
        recon, mean, log_var = model(x)
        z = mean.numpy()
        recon_np = recon.numpy().reshape(-1)
    return {
        "backend": "torch",
        "latent_shape": list(z.shape),
        "recon_samples": int(recon_np.size),
        "kl": kl_gaussian(mean.numpy(), log_var.numpy()),
    }


def torch_vae_smoke(cfg: HoliTokConfig | None = None, *, seed: int = 0) -> dict[str, Any]:
    cfg = cfg or HoliTokConfig()
    sr = cfg.sample_rate_hz
    wave = np.sin(2 * np.pi * 440 * np.arange(int(sr)) / sr)
    out = torch_encode_decode(wave, cfg=cfg, seed=seed)
    if out["backend"] == "numpy":
        return {"torch_available": False, "fallback_ok": True, **out}

    import torch

    model = build_torch_vae(cfg, seed=seed)
    hop = cfg.encoder_hop
    frames = torch.randn(4, hop)
    model.train()
    recon, mean, log_var = model(frames)
    loss = stage_ii_vae_loss(
        frames.numpy().reshape(-1)[: recon.numel()],
        recon.detach().numpy().reshape(-1),
        mean.detach().numpy(),
        log_var.detach().numpy(),
        cfg=cfg,
    )
    return {
        "torch_available": True,
        "latent_shape": list(mean.shape),
        "stage_ii_total": loss["total"],
        "backend": "torch",
    }
