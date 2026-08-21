"""QLoRA adapter specification for Qwen2.5-Omni Thinker (arXiv:2606.05713)."""

from __future__ import annotations

from typing import Any

from ltx_trainer.omni_msa.config import OmniMsaConfig


def qlora_spec(cfg: OmniMsaConfig | None = None) -> dict[str, Any]:
    """Document QLoRA hyperparameters and trainable budget (no bitsandbytes)."""
    cfg = cfg or OmniMsaConfig()
    total_params_b = 9.0  # Qwen2.5-Omni-7B Thinker order-of-magnitude
    trainable_pct = cfg.trainable_param_pct / 100.0
    return {
        "backbone": cfg.backbone,
        "quantization": {
            "bits": cfg.quant_bits,
            "dtype": cfg.quant_dtype,
            "compute_dtype": cfg.compute_dtype,
        },
        "lora": {
            "rank": cfg.lora_rank,
            "alpha": cfg.lora_alpha,
            "dropout": cfg.lora_dropout,
            "target_modules": list(cfg.lora_targets),
        },
        "trainable": {
            "params_m": cfg.trainable_params_m,
            "pct": cfg.trainable_param_pct,
            "head_only_params_k": round(cfg.head_hidden * cfg.hidden_size / 1000, 1),
        },
        "learning_rates": {"lora": cfg.lora_lr, "head": cfg.head_lr},
        "estimated_total_params_b": total_params_b,
        "estimated_trainable_fraction": round(trainable_pct, 4),
        "peak_memory_gb": f"{cfg.peak_memory_gb_min}–{cfg.peak_memory_gb_max}",
        "gpu_reference": cfg.gpu,
    }


def trainable_param_summary(cfg: OmniMsaConfig | None = None) -> str:
    spec = qlora_spec(cfg)
    return (
        f"{spec['backbone']}: {spec['trainable']['params_m']}M trainable "
        f"({spec['trainable']['pct']}% of ~{spec['estimated_total_params_b']}B), "
        f"LoRA r={spec['lora']['rank']} on {len(spec['lora']['target_modules'])} module groups"
    )
