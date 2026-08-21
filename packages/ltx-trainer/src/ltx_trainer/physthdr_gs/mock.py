"""Runnable evaluation smoke for PhysHDR-GS (arXiv:2603.28020)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.physthdr_gs.benchmarks import PAPER_ARXIV, TABLE2_EXP3, benchmarks_bundle
from ltx_trainer.physthdr_gs.metrics import PSNR_GAIN_OVER_HDR_GS


def evaluation_smoke() -> dict[str, Any]:
    ours = TABLE2_EXP3["ours_dagger"]["hdr"]
    hdr_gs = TABLE2_EXP3["hdr_gs"]["hdr"]
    out: dict[str, Any] = {
        "package": "physthdr_gs",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_syn_hdr_psnr": ours["psnr"],
        "beats_hdr_gs_hdr_psnr": ours["psnr"] > hdr_gs["psnr"],
        "psnr_gain_db": PSNR_GAIN_OVER_HDR_GS,
    }
    try:
        import torch

        from ltx_trainer.physthdr_gs.config import PhysHDRConfig
        from ltx_trainer.physthdr_gs.losses import PhysHDRLoss
        from ltx_trainer.physthdr_gs.model import PhysHDRGS
        from ltx_trainer.physthdr_gs.pipeline import train_step
        from ltx_trainer.physthdr_gs.synthetic import sample_training_pair

        from ltx_trainer.physthdr_gs.cameras import synthetic_camera_rig

        cfg = PhysHDRConfig(image_size=32, num_gaussians=64, num_train_views=4)
        model = PhysHDRGS(cfg)
        n_params = sum(p.numel() for p in model.parameters())

        target, _hdr, exp = sample_training_pair(32, seed=0)
        cam = synthetic_camera_rig(1, 32, 32, device=target.device)[0]
        t = torch.tensor(exp, dtype=target.dtype)
        with torch.no_grad():
            fwd = model(t, lighting_level=t, camera=cam)

        model.train()
        loss, stats = train_step(
            model, PhysHDRLoss(), target_ldr=target, exposure=exp, camera=cam
        )

        out.update(
            {
                "torch": True,
                "params_k": round(n_params / 1000.0, 2),
                "ihdr_mean": round(float(fwd.ihdr.mean()), 4),
                "ildr_mean": round(float(fwd.ildr.mean()), 4),
                "loss_total": round(float(loss.item()), 5),
                **{k: round(v, 5) if isinstance(v, float) else v for k, v in stats.items()},
            }
        )
        return out
    except ImportError:
        out["torch"] = False
        return out
