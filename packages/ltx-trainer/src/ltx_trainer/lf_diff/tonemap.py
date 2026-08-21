"""LF-Diff tonemap + LPR helpers (arXiv:2503.07351)."""

from __future__ import annotations

from torch import Tensor

from ltx_trainer.hdr_ingest import (
    LF_DIFF_TONEMAP_MU_DEFAULT,
    lf_diff_concat_linear_tonemap,
    lf_diff_lpr_l1,
    lf_diff_tonemap_display,
    linear_scene_to_lf_diff_tonemap_display,
)


def tonemap_for_vae(scene_linear: Tensor, *, mu: float = LF_DIFF_TONEMAP_MU_DEFAULT) -> Tensor:
    if scene_linear.dim() == 3:
        return linear_scene_to_lf_diff_tonemap_display(scene_linear.unsqueeze(0), mu=mu).squeeze(0)
    return linear_scene_to_lf_diff_tonemap_display(scene_linear, mu=mu)


def tonemap_l1(pred_linear: Tensor, target_linear: Tensor, *, mu: float = LF_DIFF_TONEMAP_MU_DEFAULT) -> Tensor:
    pred_t = tonemap_for_vae(pred_linear, mu=mu)
    tgt_t = tonemap_for_vae(target_linear, mu=mu)
    return (pred_t - tgt_t).abs().mean()


def lpr_l1(pred_lpr: Tensor, target_lpr: Tensor) -> Tensor:
    return lf_diff_lpr_l1(pred_lpr, target_lpr)


def concat_tonemap_features(hdr: Tensor, *, mu: float = LF_DIFF_TONEMAP_MU_DEFAULT) -> Tensor:
    return lf_diff_concat_linear_tonemap(hdr, mu=mu)
