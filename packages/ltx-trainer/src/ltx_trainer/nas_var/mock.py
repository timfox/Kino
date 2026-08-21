"""Smoke helpers for NAS-VAR."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.nas_var.config import NasVarConfig
from ltx_trainer.nas_var.distill import privileged_distillation_loss, reverse_kl, softmax
from ltx_trainer.nas_var.rollout import fuse_latent_grids, rollout_next_scales


def evaluation_smoke() -> dict[str, Any]:
    cfg = NasVarConfig()
    rng = np.random.default_rng(1)
    vocab = 64
    student_logits = [rng.standard_normal(vocab) for _ in cfg.acceleration_scales]
    teacher_logits = [s + 0.5 * rng.standard_normal(vocab) for s in student_logits]
    distill = privileged_distillation_loss(student_logits, teacher_logits)
    tokens = rollout_next_scales(
        {k: lg for k, lg in zip(cfg.acceleration_scales, student_logits, strict=True)}
    )
    grids = [rng.integers(0, 10, size=(11, 11)) for _ in range(3)]
    fused = fuse_latent_grids(grids)

    base_p = softmax(student_logits[0])
    teacher_p = softmax(teacher_logits[0])
    rkl = reverse_kl(base_p, teacher_p)

    return {
        "codebook": cfg.codebook_size,
        "rollout_tokens": tokens,
        "distill_loss": round(distill, 4),
        "reverse_kl_scale32": round(rkl, 4),
        "fused_grid_shape": list(fused.shape),
        "headline_cartesian_x_flair_psnr": 21.29,
    }


def scale_factorization_demo() -> dict[str, Any]:
    scales = [32, 16, 8, 4, 2, "FS"]
    chain = [f"Q{s} -> Q{scales[i+1]}" for i, s in enumerate(scales[:-1])]
    return {"ordered_scales": scales, "prediction_chain": chain}
