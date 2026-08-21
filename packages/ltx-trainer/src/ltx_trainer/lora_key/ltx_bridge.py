"""LTX native LoRA: attach reusable Watermark LoRA via linear superposition."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from ltx_trainer.lora_key.config import LoRAKeyConfig
from ltx_trainer.lora_key.lora_math import merge_loras


class LoRAKeyLTXBridge(nn.Module):
    """
    Conceptual bridge: merge creator style LoRA with frozen Watermark LoRA at load time.
    Θ_deploy = Θ0 + α∆Θ_style + γ∆Θ_key (Eq. 14).
    """

    def __init__(self, cfg: LoRAKeyConfig | None = None, *, d_in: int = 4096, d_out: int = 4096) -> None:
        super().__init__()
        cfg = cfg or LoRAKeyConfig()
        self.cfg = cfg
        r = cfg.watermark_lora_rank
        self.w0 = nn.Parameter(torch.randn(d_out, d_in) * 0.02, requires_grad=False)
        self.style_b = nn.Parameter(torch.zeros(d_out, r))
        self.style_a = nn.Parameter(torch.zeros(r, d_in))
        self.key_b = nn.Parameter(torch.zeros(d_out, r))
        self.key_a = nn.Parameter(torch.zeros(r, d_in))

    def merged_weight(self) -> Tensor:
        return merge_loras(
            self.w0,
            self.style_b,
            self.style_a,
            self.key_b,
            self.key_a,
            alpha=self.cfg.alpha_style,
            gamma=self.cfg.gamma_watermark,
        )

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.linear(x, self.merged_weight())


def ltx_integration_notes(cfg: LoRAKeyConfig | None = None) -> dict[str, Any]:
    cfg = cfg or LoRAKeyConfig()
    return {
        "use_case": "Protect exported LTX LoRA checkpoints before Civitai/HF distribution",
        "merge_at": "inference.py load_lora_weights — add γ·ΔΘ_key to each targeted linear",
        "no_retrain": "Style LoRA unchanged; one Watermark LoRA per creator",
        "verify": "Decode 48-bit message from suspicious frame latents via frozen D_φ",
        "rank": cfg.watermark_lora_rank,
        "pair_with": ["kino-ltx-quality-gate", "kino-delivery-preflight"],
    }
