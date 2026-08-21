"""Runnable evaluation smoke for Gimbal360 (arXiv:2603.23179)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.gimbal360.benchmarks import TABLE1_METRICS, benchmarks_bundle
from ltx_trainer.gimbal360.config import PAPER_ARXIV


def evaluation_smoke() -> dict[str, Any]:
    ours = next(r for r in TABLE1_METRICS if r["method"] == "Ours")
    out: dict[str, Any] = {
        "package": "gimbal360",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_indoor_fid": ours["indoor"]["FID"],
        "ref_outdoor_cs": ours["outdoor"]["CS"],
    }
    try:
        import torch

        from ltx_trainer.gimbal360.auto_leveling import DifferentiableAutoLeveling
        from ltx_trainer.gimbal360.canonical import center_erp_by_yaw
        from ltx_trainer.gimbal360.config import Gimbal360Config
        from ltx_trainer.gimbal360.gimbal360_net import Gimbal360CompletionStub
        from ltx_trainer.gimbal360.pipeline import evaluation_demo_run
        from ltx_trainer.gimbal360.circular_vae import CircularVAEStub
        from ltx_trainer.gimbal360.fill_conditioning import concat_fill_channels, fill_channel_count
        from ltx_trainer.gimbal360.horizon360 import dataset_card
        from ltx_trainer.gimbal360.inference import SamplerConfig, shift_equivariant_sample
        from ltx_trainer.gimbal360.topology import roll_azimuth, siamese_shift_loss

        cfg = Gimbal360Config(erp_height=48, erp_width=96, perspective_height=32, perspective_width=32)
        demo = evaluation_demo_run(cfg, device="cpu")
        erp = torch.rand(1, 3, 48, 96)
        centered = center_erp_by_yaw(erp, 12)
        dal = DifferentiableAutoLeveling()
        flow, rot, zc = dal(torch.rand(1, 3, 32, 32))
        model = Gimbal360CompletionStub(cfg)
        batch_p = torch.rand(1, 3, 32, 32)
        batch_m = torch.ones(1, 1, 12, 24)
        batch_z = torch.randn(1, 4, 12, 24)
        o = model(batch_p, batch_m, batch_z, shift_delta=3)
        l_shift = siamese_shift_loss(o["eps_base"], o["eps_shifted"], 3)
        z = torch.randn(1, cfg.latent_channels, 12, 24)
        m = torch.ones(1, 1, 12, 24)
        m[..., 8:] = 0
        samp = shift_equivariant_sample(
            model, batch_p, m, z, sampler=SamplerConfig(num_steps=3, step_size=0.1)
        )
        vae = CircularVAEStub(cfg.latent_channels)
        enc = vae.encode(batch_p)
        fill_ch = fill_channel_count(cfg.latent_channels)
        out.update(
            {
                "torch": True,
                "horizon360_size": dataset_card()["size"],
                "fill_channels": fill_ch,
                "centered_shape": list(centered.shape),
                "flow_finite": bool(torch.isfinite(flow).all()),
                "rot_det": float(torch.det(rot[0]).item()),
                "siamese_loss": float(l_shift.item()),
                "demo_loss": demo["train"]["loss"],
                "infer_shift": int(samp["total_azimuth_shift"].item()),
                "vae_encode_shape": list(enc.shape),
            }
        )
        return out
    except ImportError:
        out["torch"] = False
        return out
