"""Runnable evaluation smoke for VDP-HDR (arXiv:2605.11628)."""

from __future__ import annotations

from typing import Any


def evaluation_smoke(*, seed: int = 0) -> dict[str, Any]:
    out: dict[str, Any] = {"package": "vdp_hdr", "paper": "arXiv:2605.11628"}
    try:
        import torch

        from ltx_trainer.vdp_hdr.metrics import q_mae, q_psnr
        from ltx_trainer.vdp_hdr.model import VdpHdr, VdpHdrConfig
        from ltx_trainer.vdp_hdr.pipeline import recover_hdr_from_ldr, train_step
        from ltx_trainer.vdp_hdr.synthetic import synthesize_hdr_pair

        torch.manual_seed(seed)
        cfg = VdpHdrConfig()
        model = VdpHdr(cfg)
        model.train()

        hdr_gt, ldr, bracket = synthesize_hdr_pair(48, 48)
        loss, stats = train_step(model, ldr=ldr, hdr=hdr_gt.unsqueeze(0), bracket=bracket.unsqueeze(0))

        model.eval()
        with torch.no_grad():
            hdr_rec = recover_hdr_from_ldr(model, ldr)

        n_params = sum(p.numel() for p in model.parameters())
        out.update(
            {
                "torch": True,
                "params_k": round(n_params / 1000.0, 2),
                "loss_fusion": round(float(loss.item()), 5),
                "q_psnr": round(stats["q_psnr"], 3),
                "q_mae": round(stats["q_mae"], 4),
                "bracket_frames": cfg.num_frames,
                "hdr_finite": bool(torch.isfinite(hdr_rec).all()),
                "recover_psnr": round(float(q_psnr(hdr_rec, hdr_gt)), 3),
            }
        )
        return out
    except ImportError:
        import numpy as np

        x = np.linspace(0, 1, 16, dtype=np.float64)
        y = np.clip(x * 1.2, 0, 1)
        out.update({"torch": False, "tone_mae": round(float(np.abs(x - y).mean()), 4)})
        return out
