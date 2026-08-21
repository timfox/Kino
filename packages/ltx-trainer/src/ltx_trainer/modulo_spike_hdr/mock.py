"""Runnable evaluation smoke for modulo_spike_hdr."""

from __future__ import annotations

from typing import Any

from ltx_trainer.modulo_spike_hdr.benchmarks import PAPER_ARXIV, TABLE1_SYNTHETIC, benchmarks_bundle


def evaluation_smoke() -> dict[str, Any]:
    out: dict[str, Any] = {
        "package": "modulo_spike_hdr",
        "paper": f"arXiv:{PAPER_ARXIV}",
        "benchmarks": benchmarks_bundle(),
        "ref_psnr_l": TABLE1_SYNTHETIC["ours"]["psnr_l"],
        "ref_per_frame_s": TABLE1_SYNTHETIC["ours"]["per_frame_s"],
    }
    try:
        import torch

        from ltx_trainer.modulo_spike_hdr.lar import lar_gradient_features, modulo_image
        from ltx_trainer.modulo_spike_hdr.losses import UnwrapLoss
        from ltx_trainer.modulo_spike_hdr.model import ModuloSpikeHdrUnwrapper
        from ltx_trainer.modulo_spike_hdr.spike_encode import (
            encode_modulo_from_spikes,
            exposure_decoupled_fps,
            synthesize_spike_frames,
        )
        from ltx_trainer.modulo_spike_hdr.synthetic import synthesize_modulo_pair
        from ltx_trainer.lucky_hdr.tonemap import tone_map_mu

        period = 256.0
        hdr = torch.rand(3, 48, 48) * 0.8 + 0.05
        modulo, linear, i_mu_gt = synthesize_modulo_pair(hdr, period=period)

        model = ModuloSpikeHdrUnwrapper()
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())

        with torch.no_grad():
            i_mu, i_lin, _ = model(modulo)

        spikes = synthesize_spike_frames(hdr, num_frames=80)
        encoded = encode_modulo_from_spikes(spikes, window=25, stride=20, period=period)
        fps = exposure_decoupled_fps(readout_hz=20_000.0, stride_frames=20)

        grad = lar_gradient_features(modulo.unsqueeze(0), period)
        wrapped = modulo_image(i_lin * period, period) / period
        wrap_mae = float((wrapped - modulo).abs().mean())

        loss_fn = UnwrapLoss()
        loss, stats = loss_fn(i_mu, i_lin, i_mu_gt, hdr, modulo * period)

        out.update(
            {
                "torch": True,
                "params_k": round(n_params / 1000.0, 2),
                "spike_frames": int(spikes.shape[0]),
                "modulo_queries": len(encoded),
                "effective_fps": round(fps, 1),
                "lar_grad_channels": int(grad.shape[1]),
                "wrap_consistency_mae": round(wrap_mae, 5),
                "loss_total": round(float(loss.item()), 5),
                "loss_phys": stats["loss_phys"],
            }
        )
        return out
    except ImportError:
        out["torch"] = False
        return out
