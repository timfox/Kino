"""HoliTok evaluation smoke (arXiv:2605.29948)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.holitok.config import HoliTokConfig
from ltx_trainer.holitok.pipeline import pipeline_demo
from ltx_trainer.holitok.vae import compression_ratio


def evaluation_smoke(cfg: HoliTokConfig | None = None) -> dict[str, Any]:
    c = cfg or HoliTokConfig()
    demo = pipeline_demo(c, seed=0)
    cr = compression_ratio(c.sample_rate_hz, c.latent_frame_rate_hz, c.latent_dim, bfloat_bits=c.bfloat_bits)
    return {
        "paper": c.paper_arxiv,
        "compression_ratio": round(cr, 2),
        "latent_hz": c.latent_frame_rate_hz,
        "latent_dim": c.latent_dim,
        "stage_iii_total": demo["stage_iii_total"],
        "generation_total": demo["generation_total"],
        "understanding_ce": demo["understanding_ce"],
        "vae_encode_shape_ok": demo["vae_encode_shape_ok"],
        "sampler_generation_finite": bool(demo["sampler_generation_finite"]),
        "torch_vae_backend": demo["torch_vae_backend"],
        "stage_ii_loss_decreased": demo["stage_ii_loss_decreased"],
    }
