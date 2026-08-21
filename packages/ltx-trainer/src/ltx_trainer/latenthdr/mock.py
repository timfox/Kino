"""Runnable evaluation smoke for LatentHDR (arXiv:2605.11115)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.latenthdr.benchmarks import PAPER_ARXIV, TABLE2_SI_HDR, benchmarks_bundle

_LATENT_CHANNELS = 16


def evaluation_smoke() -> dict[str, Any]:
    """CPU/torch smoke: EV head, l2h bracket, log merge, dynamic range."""
    out: dict[str, Any] = {
        "package": "latenthdr",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_si_hdr_stops_v2": TABLE2_SI_HDR["ours_v2_stops"],
    }
    try:
        import torch

        from ltx_trainer.hdr_ingest import ev_list_arange, merge_log_domain_radiance
        from ltx_trainer.latenthdr.exposure_head import FiLMResidualExposureHead, exposure_latent_mse
        from ltx_trainer.latenthdr.metrics import dynamic_range_stops
        from ltx_trainer.latenthdr.model import LatentHdr, LatentHdrConfig
        from ltx_trainer.latenthdr.synthetic import synthesize_hdr_scene, synthesize_ldr_from_hdr

        channels = _LATENT_CHANNELS
        z = torch.randn(2, channels, 4, 8, 8)
        ev = torch.tensor([0.0, 1.5])
        head = FiLMResidualExposureHead(latent_channels=channels)
        z_hat = head(z, ev)
        mse = exposure_latent_mse(z_hat, z + 0.02 * torch.randn_like(z))
        out.update(
            {
                "latent_channels": channels,
                "ev_mse": round(float(mse.item()), 5),
                "stack_brackets": 2,
            }
        )

        cfg = LatentHdrConfig(ev_min=-3.0, ev_max=3.0, ev_step=1.0, head_type="unet")
        model = LatentHdr(cfg)
        hdr_gt = synthesize_hdr_scene(48, 48)
        ldr = synthesize_ldr_from_hdr(hdr_gt, ev=0.0)
        hdr_rec = model(ldr)
        stops = dynamic_range_stops(hdr_rec)
        out["l2h_dynamic_range_stops"] = round(stops, 3)
        out["l2h_finite"] = bool(torch.isfinite(hdr_rec).all().item())

        evs = ev_list_arange(-2.0, 2.0, 1.0)
        stack = torch.stack([synthesize_ldr_from_hdr(hdr_gt, ev=e) for e in evs], dim=0).unsqueeze(2)
        merged = merge_log_domain_radiance(stack, evs, gamma=cfg.gamma).squeeze(1)
        out["merge_stops"] = round(dynamic_range_stops(merged), 3)
        out["n_exposures_merged"] = len(evs)
        return out
    except ImportError:
        z = np.random.default_rng(0).standard_normal((channels := _LATENT_CHANNELS, 4, 8, 8))
        mse = float(np.mean((z - (z + 0.01)) ** 2))
        out.update({"latent_channels": channels, "ev_mse_proxy": round(mse, 5), "torch": False})
        return out
