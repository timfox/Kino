"""Composite training objective — Appendix C.1."""

from __future__ import annotations

from typing import Any

import numpy as np

from ltx_trainer.fctts.ccl import conditional_consistency_loss
from ltx_trainer.fctts.cfm import cfm_loss
from ltx_trainer.fctts.config import FcttsConfig


def blur_mae_loss(h: np.ndarray, x0: np.ndarray) -> float:
    """L_blur = E[||h - x0||] — §3.2.1."""
    return float(np.mean(np.abs(np.asarray(h) - np.asarray(x0))))


def total_loss_components(
    x0: np.ndarray,
    x1: np.ndarray,
    h_blur: np.ndarray,
    u_pred: np.ndarray,
    cp: np.ndarray,
    z_spk: np.ndarray,
    prosody_logits: np.ndarray,
    spk_pred: np.ndarray,
    cfg: FcttsConfig | None = None,
) -> dict[str, Any]:
    cfg = cfg or FcttsConfig()
    l_cfm = cfm_loss(u_pred, x0, x1)
    l_blur = blur_mae_loss(h_blur, x0)
    ccl = conditional_consistency_loss(
        x0,
        cp,
        z_spk,
        prosody_logits,
        spk_pred,
        lambda_pro=cfg.lambda_ccl_pro,
        lambda_spk=cfg.lambda_ccl_spk,
    )
    l_total = (
        cfg.lambda_cfm * l_cfm
        + cfg.lambda_blur * l_blur
        + ccl["L_ccl"]
    )
    return {
        "L_total": l_total,
        "L_cfm": l_cfm,
        "L_blur": l_blur,
        **ccl,
    }
