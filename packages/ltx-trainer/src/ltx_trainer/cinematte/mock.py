"""Runnable evaluation smoke for CineMatte (arXiv:2605.18328)."""

from __future__ import annotations

from typing import Any

import torch

from ltx_trainer.cinematte.losses import CineMatteLoss
from ltx_trainer.cinematte.metrics import dtssd, gradient_error, matting_mad, matting_mse
from ltx_trainer.cinematte.model import CineMatte, CineMatteConfig
from ltx_trainer.cinematte.pipeline import apply_background_shift, matte_image
from ltx_trainer.cinematte.synthetic import composite_training_sample


def evaluation_smoke() -> dict[str, Any]:
    device = "cpu"
    model = CineMatte(CineMatteConfig(backbone="stub", embed_dim=64, fbam_layers=2))
    loss_fn = CineMatteLoss()
    img, bg, alpha = composite_training_sample(64, 64, device=torch.device(device))
    result = matte_image(model, img, bg)
    pred = model(img.unsqueeze(0), bg.unsqueeze(0)).squeeze(1)
    loss, stats = loss_fn(pred, alpha.unsqueeze(0))
    shifted = apply_background_shift(bg, angle_deg=4.0, scale=1.08, shear=0.05)
    seq_p = torch.stack([pred.squeeze(0), pred.squeeze(0) * 0.95])
    seq_g = torch.stack([alpha.squeeze(0), alpha.squeeze(0)])
    return {
        "package": "cinematte",
        "paper": "arXiv:2605.18328",
        "alpha_shape": list(result.alpha.shape),
        "loss_total": round(float(stats["loss_total"]), 4),
        "mad": round(matting_mad(pred, alpha.unsqueeze(0)), 3),
        "mse": round(matting_mse(pred, alpha.unsqueeze(0)), 3),
        "grad": round(gradient_error(pred, alpha.unsqueeze(0)), 3),
        "dtssd": round(dtssd(seq_p, seq_g), 3),
        "shift_delta": round(float((shifted - bg).abs().mean()), 4),
    }
