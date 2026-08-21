"""Runnable evaluation smoke for LumaFlux (arXiv:2604.02787)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lumaflux.benchmarks import (
    PAPER_ARXIV,
    TABLE1_BENCHMARKS,
    benchmarks_bundle,
)


def evaluation_smoke() -> dict[str, Any]:
    lf = TABLE1_BENCHMARKS["luma_eval"]["lumaflux"]
    base = TABLE1_BENCHMARKS["luma_eval"]["hdrtvnet++"]
    out: dict[str, Any] = {
        "package": "lumaflux",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_luma_eval_psnr": lf["psnr"],
        "ref_luma_eval_delta_e_itp": lf["delta_e_itp"],
        "beats_hdrtvnet_on_psnr": lf["psnr"] > base["psnr"],
    }
    try:
        import torch

        from ltx_trainer.lumaflux.config import LumaFluxConfig
        from ltx_trainer.lumaflux.losses import LumaFluxLoss
        from ltx_trainer.lumaflux.model import LumaFlux
        from ltx_trainer.lumaflux.pipeline import train_step
        from ltx_trainer.lumaflux.synthetic import synthesize_pair

        cfg = LumaFluxConfig(image_size=32, num_blocks=2)
        model = LumaFlux(cfg)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())

        sdr, hdr = synthesize_pair(32, seed=0)
        with torch.no_grad():
            pred = model(sdr, t=0.5).hdr

        model.train()
        loss, stats = train_step(model, LumaFluxLoss(), sdr=sdr, hdr_gt=hdr)

        out.update(
            {
                "torch": True,
                "params_k": round(n_params / 1000.0, 2),
                "pred_hdr_max": round(float(pred.max()), 4),
                "loss_total": round(float(loss.item()), 5),
                **{k: round(v, 5) if isinstance(v, float) else v for k, v in stats.items()},
            }
        )
        return out
    except ImportError:
        out["torch"] = False
        return out
