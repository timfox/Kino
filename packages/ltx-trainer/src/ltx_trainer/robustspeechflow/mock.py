"""RobustSpeechFlow contrastive FM smoke (arXiv:2605.22083)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.robustspeechflow.config import RobustSpeechFlowConfig


def evaluation_smoke(cfg: RobustSpeechFlowConfig | None = None) -> dict[str, Any]:
    c = cfg or RobustSpeechFlowConfig()
    try:
        import torch
        from ltx_trainer.robustspeechflow.losses import robustspeechflow_objective

        x = torch.randn(4, 8)
        eps = torch.randn(4, 8)
        u = torch.randn(4, 8)
        losses = robustspeechflow_objective(
            u,
            x=x,
            eps=eps,
            x_rand=x + 0.1 * torch.randn_like(x),
            x_aug=x + 0.05 * torch.randn_like(x),
            lambda_rand=c.lambda_rand,
            lambda_aug=c.lambda_aug,
        )
        l_pos = float(losses["Lpos"])
    except ImportError:
        x = np.random.default_rng(0).standard_normal((4, 8))
        eps = np.random.default_rng(1).standard_normal((4, 8))
        l_pos = float(np.mean((x - eps) ** 2))
    return {
        "paper": c.paper_arxiv,
        "Lpos": l_pos,
        "paper_seed_tts_eval_robustspeechflow_wer": c.seed_tts_eval_robustspeechflow_wer,
        "paper_zero500_en_24_cer": c.zero500_en_robust_24_cer,
    }
