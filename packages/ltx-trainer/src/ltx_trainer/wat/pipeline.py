"""Framework card, tables, evaluation demo."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ltx_trainer.wat.config import WATConfig
from ltx_trainer.wat.haar import haar_dwt, haar_idwt
from ltx_trainer.wat.model import WATSharedStub
from ltx_trainer.wat.schema import TokenBatch, coeffs_to_tokens, tokens_to_coeffs
from ltx_trainer.wat.selection import apply_mask, psnr_from_mse, select_topk_energy


def framework_card(cfg: WATConfig | None = None) -> dict[str, Any]:
    cfg = cfg or WATConfig()
    return {
        "paper": cfg.paper_arxiv,
        "title": cfg.title,
        "author": cfg.author,
        "paradigm": "one-level Haar DWT + shared coefficient token schema",
        "modalities": {
            "audio": {"dataset": cfg.audio_dataset, "dense_tokens": cfg.dense_tokens_audio},
            "image": {"dataset": cfg.image_dataset, "dense_tokens": cfg.dense_tokens_image},
            "video": {"dataset": cfg.video_dataset, "dense_tokens": cfg.dense_tokens_video},
        },
        "token_fields": ["value", "modality", "rank", "scale", "subband", "position"],
        "audio_scale": cfg.audio_scale,
        "latent_dim": cfg.latent_dim,
        "note": "continuous tokens only; no FSQ/VQ in this study",
    }


def table1_dense_validation() -> list[dict[str, Any]]:
    """Table 1 — preliminary validation PSNR."""
    return [
        {"model": "Separate baseline", "A-PSNR": 40.79, "I-PSNR": 22.91, "V-PSNR": 20.91},
        {"model": "Shared schema, unit scale", "A-PSNR": 26.30, "I-PSNR": 30.15, "V-PSNR": 22.53},
        {
            "model": "Shared schema, audio scale 4",
            "A-PSNR": 39.92,
            "I-PSNR": 29.37,
            "V-PSNR": 23.93,
        },
    ]


def table2_matched_latent_16() -> list[dict[str, Any]]:
    """Table 2 — matched continuous latent scalar budget at latent dim 16."""
    return [
        {"model": "Separate matched", "Audio": 43.93, "Image": 22.47, "Video": 21.76},
        {"model": "Shared full metadata", "Audio": 39.92, "Image": 29.37, "Video": 23.93},
        {"model": "Shared no metadata", "Audio": 41.29, "Image": 30.41, "Video": 29.42},
    ]


def table4_energy_gains() -> list[dict[str, Any]]:
    """Table 4 — energy global vs uniform/random (average dB gain)."""
    return [
        {"modality": "Audio", "vs_uniform": 16.73, "vs_random": 16.73},
        {"modality": "Image", "vs_uniform": 16.90, "vs_random": 16.92},
        {"modality": "Video", "vs_uniform": 15.86, "vs_random": 15.85},
    ]


def table5_masked_sparse() -> list[dict[str, Any]]:
    """Table 5 — best masked sparse shared results."""
    return [
        {"modality": "Audio", "metadata": "none", "latent": 8, "keep": 0.50, "PSNR": 32.60},
        {"modality": "Image", "metadata": "full", "latent": 8, "keep": 0.50, "PSNR": 29.98},
        {"modality": "Video", "metadata": "none", "latent": 16, "keep": 0.50, "PSNR": 34.45},
    ]


def _demo_audio(cfg: WATConfig) -> tuple[Tensor, tuple[int, ...], str]:
    t = 256
    x = torch.randn(cfg.demo_batch, 1, t) * 0.1
    coeffs = haar_dwt(x, 1)
    grid = (coeffs.shape[-1],)
    return x, grid, "audio"


def _demo_image(cfg: WATConfig) -> tuple[Tensor, tuple[int, ...], str]:
    x = torch.rand(cfg.demo_batch, 3, 16, 16)
    coeffs = haar_dwt(x, 2)
    grid = coeffs.shape[-2:]
    return x, grid, "image"


def _demo_video(cfg: WATConfig) -> tuple[Tensor, tuple[int, ...], str]:
    x = torch.rand(cfg.demo_batch, 3, 4, 16, 16)
    coeffs = haar_dwt(x, 3)
    grid = coeffs.shape[-3:]
    return x, grid, "video"


def schema_roundtrip_error(x: Tensor, modality: str) -> float:
    coeffs = haar_dwt(x, {"audio": 1, "image": 2, "video": 3}[modality])
    rank = {"audio": 1, "image": 2, "video": 3}[modality]
    grid = coeffs.shape[2:]
    batch = coeffs_to_tokens(coeffs, modality)  # type: ignore[arg-type]
    back = tokens_to_coeffs(batch, modality, grid)  # type: ignore[arg-type]
    c = x.shape[1]
    recon = haar_idwt(back, rank, c)
    recon = recon[..., : x.shape[-1]] if rank == 1 else recon
    if rank == 2:
        recon = recon[..., : x.shape[-2], : x.shape[-1]]
    if rank == 3:
        recon = recon[..., : x.shape[-3], : x.shape[-2], : x.shape[-1]]
    return float((coeffs - back).abs().max())


def forward_smoke(cfg: WATConfig | None = None) -> dict[str, Any]:
    cfg = cfg or WATConfig()
    model = WATSharedStub(cfg)
    results: dict[str, Any] = {}
    for name, factory in (
        ("audio", _demo_audio),
        ("image", _demo_image),
        ("video", _demo_video),
    ):
        x, grid, mod = factory(cfg)
        rank = {"audio": 1, "image": 2, "video": 3}[mod]
        coeffs = haar_dwt(x, rank)
        batch = coeffs_to_tokens(coeffs, mod)  # type: ignore[arg-type]
        pred = model.forward_tokens(batch, mod, use_metadata=False)
        batch_hat = TokenBatch(
            pred,
            batch.modality,
            batch.rank,
            batch.scale,
            batch.subband,
            batch.position,
        )
        coeff_hat = tokens_to_coeffs(batch_hat, mod, grid)  # type: ignore[arg-type]
        x_hat = haar_idwt(coeff_hat, rank, x.shape[1])
        if rank == 1:
            x_hat = x_hat[..., : x.shape[-1]]
        elif rank == 2:
            x_hat = x_hat[..., : x.shape[-2], : x.shape[-1]]
        else:
            x_hat = x_hat[..., : x.shape[-3], : x.shape[-2], : x.shape[-1]]
        mse = float(F.mse_loss(x_hat, x).item())
        masked = apply_mask(batch.values, select_topk_energy(batch.values, 0.5))
        results[mod] = {
            "schema_max_err": schema_roundtrip_error(x, mod),
            "recon_mse": mse,
            "recon_psnr": psnr_from_mse(mse),
            "token_count": batch.values.shape[1],
            "masked_keep_50": int(masked.abs().sum(dim=-1).gt(0).sum(dim=-1).float().mean()),
        }
    return results


def evaluation_demo(cfg: WATConfig | None = None) -> dict[str, Any]:
    cfg = cfg or WATConfig()
    row = next(r for r in table1_dense_validation() if "audio scale 4" in r["model"])
    return {
        "framework": framework_card(cfg),
        "table1": table1_dense_validation(),
        "table2": table2_matched_latent_16(),
        "table4": table4_energy_gains(),
        "table5": table5_masked_sparse(),
        "headline": {
            "dense_audio_psnr": row["A-PSNR"],
            "dense_image_psnr": row["I-PSNR"],
            "dense_video_psnr": row["V-PSNR"],
            "masked_video_psnr": cfg.masked_video_psnr,
        },
        "forward": forward_smoke(cfg),
    }
