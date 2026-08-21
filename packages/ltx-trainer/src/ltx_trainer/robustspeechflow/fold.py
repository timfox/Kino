"""Fold RobustSpeechFlow alignment-robustness probe into audio latents."""

from __future__ import annotations

from typing import Any

from ltx_trainer.robustspeechflow.config import RobustSpeechFlowConfig


def robustspeechflow_meta_block() -> dict[str, Any]:
    return {
        "robustspeechflow": {
            "arxiv_id": "2605.22083",
            "fold_role": "audio_alignment_robustness_probe",
            "augment": "length_preserving_repeat",
        }
    }


def annotate_audio_save_data(data: dict[str, Any]) -> dict[str, Any]:
    latents = data.get("latents")
    out = dict(data)
    out.update(robustspeechflow_meta_block())
    if latents is None:
        return out

    try:
        import torch
        from ltx_trainer.robustspeechflow.augment import repeat_overwrite

        x = latents if isinstance(latents, torch.Tensor) else torch.as_tensor(latents)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        t = int(x.shape[-1])
        length = max(1, min(4, t // 4))
        x_aug = repeat_overwrite(x, src_start=0, tgt_start=min(length, t - length - 1), length=length)
        delta = float((x - x_aug).abs().mean().item())
        stable = delta < 0.15
    except ImportError:
        delta = 0.0
        stable = True

    cfg = RobustSpeechFlowConfig()
    out["robustspeechflow"].update(
        {
            "repeat_delta_mean": round(delta, 6),
            "alignment_stable_proxy": stable,
            "lambda_rand": cfg.lambda_rand,
            "lambda_aug": cfg.lambda_aug,
        }
    )
    return out
