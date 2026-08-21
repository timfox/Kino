"""Runnable evaluation smoke for LumiVid (arXiv:2604.11788)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.lumivid.benchmarks import PAPER_ARXIV, TABLE1_VAE_ROUNDTRIP, TABLE2_BASELINES_ARRI, benchmarks_bundle


def evaluation_smoke() -> dict[str, Any]:
    out: dict[str, Any] = {
        "package": "lumivid",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_arri_pu21_psnr": TABLE2_BASELINES_ARRI["lumivid"]["pu21_psnr"],
        "ref_logc3_kl_lat": TABLE1_VAE_ROUNDTRIP["logc3"]["kl_lat"],
    }
    try:
        import torch

        from ltx_trainer.hdr_ingest import lumivid_meta_block
        from ltx_trainer.lumivid.encodings import HdrEncoding, encode_hdr, kl_histogram_proxy
        from ltx_trainer.lumivid.logc3_codec import scene_linear_to_vae_pixels, vae_pixels_to_scene_linear
        from ltx_trainer.lumivid.model import LumiVid
        from ltx_trainer.lumivid.synthetic import synthesize_hdr_scene, synthesize_sdr_from_hdr
        from ltx_trainer.lumivid.vae_stub import VaeStub

        hdr = synthesize_hdr_scene(48, 48)
        sdr = synthesize_sdr_from_hdr(hdr)
        model = LumiVid()
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())

        with torch.no_grad():
            pred_hdr = model.infer(sdr.unsqueeze(0), steps=3).squeeze(0)

        logc3 = scene_linear_to_vae_pixels(hdr)
        roundtrip = vae_pixels_to_scene_linear(logc3)
        logc3_mae = float((roundtrip - hdr.clamp(min=0.0)).abs().mean())

        sdr_ref = torch.rand(3, 32, 32).clamp(0, 1)
        sdr_dist = sdr_ref.clamp(0, 1)
        kl_logc3 = kl_histogram_proxy(encode_hdr(hdr, HdrEncoding.LOGC3), sdr_dist)

        vae = VaeStub(latent_ch=16)
        sdr_enc = sdr_dist.unsqueeze(0) * 2 - 1
        z = vae.encode(sdr_enc)
        recon = ((vae.decode(z) + 1) * 0.5).clamp(0, 1)
        kl_lat = kl_histogram_proxy(z.flatten(), torch.randn_like(z).flatten() * 0.01)

        model.train()
        loss, stats = model.training_step(sdr, hdr)

        meta = lumivid_meta_block()
        out.update(
            {
                "torch": True,
                "params_k": round(n_params / 1000.0, 2),
                "pred_hdr_max": round(float(pred_hdr.max()), 4),
                "logc3_roundtrip_mae": round(logc3_mae, 5),
                "kl_px_logc3_proxy": round(kl_logc3, 4),
                "loss_total": round(float(loss.item()), 5),
                "loss_flow": stats["loss_flow"],
                "lumivid_meta_arxiv": meta["lumivid"]["arxiv_id"],
            }
        )
        return out
    except ImportError:
        out["torch"] = False
        return out
