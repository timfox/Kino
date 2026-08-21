"""Runnable evaluation smoke for DiffHDR (arXiv:2604.06161)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.diffhdr.benchmarks import (
    PAPER_ARXIV,
    TABLE1_SI_HDR,
    TABLE4_LOG_GAMMA,
    benchmarks_bundle,
)


def evaluation_smoke() -> dict[str, Any]:
    out: dict[str, Any] = {
        "package": "diffhdr",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_si_hdr_pu21_piqe": TABLE1_SI_HDR["ours"]["pu21_piqe"],
        "ref_log_gamma_psnr": TABLE4_LOG_GAMMA["ours"]["psnr"],
    }
    try:
        import torch

        from ltx_trainer.diffhdr.config import DiffHDRConfig
        from ltx_trainer.diffhdr.model import DiffHDR
        from ltx_trainer.diffhdr.pipeline import train_step
        from ltx_trainer.diffhdr.synthetic import synthesize_pair, vae_roundtrip_error

        cfg = DiffHDRConfig(num_frames=4, image_size=32)
        model = DiffHDR(cfg)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())

        ldr, hdr = synthesize_pair(4, 32, seed=0)
        with torch.no_grad():
            pred = model(ldr, hdr_target=None).hdr

        err_lg = float(vae_roundtrip_error(hdr[0], mapping="log_gamma"))
        err_lin = float(vae_roundtrip_error(hdr[0], mapping="linear"))

        model.train()
        loss, stats = train_step(model, ldr=ldr, hdr=hdr)

        out.update(
            {
                "torch": True,
                "params_k": round(n_params / 1000.0, 2),
                "pred_hdr_max": round(float(pred.max()), 4),
                "vae_roundtrip_log_gamma": round(err_lg, 5),
                "vae_roundtrip_linear": round(err_lin, 5),
                "log_gamma_beats_linear": err_lg < err_lin,
                "loss_flow": round(float(loss.item()), 5),
                **{k: round(v, 5) if isinstance(v, float) else v for k, v in stats.items()},
            }
        )
        return out
    except ImportError:
        out["torch"] = False
        return out
