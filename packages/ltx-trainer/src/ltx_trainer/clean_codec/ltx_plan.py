"""LTX / AV fold hooks for CleanCodec tokens."""

from __future__ import annotations

from typing import Any

from ltx_trainer.clean_codec.config import CleanCodecConfig


def ltx_integration_plan(cfg: CleanCodecConfig | None = None) -> dict[str, Any]:
    cfg = cfg or CleanCodecConfig()
    return {
        "paper": cfg.paper_arxiv,
        "use_cases": [
            "12.5 t/s speech tokens for LTX audio sidecars (vs 40–200 t/s baselines)",
            "Global speaker emb + local tokens for VC/TTS conditioning in teleplay pipeline",
            "Denoising codec front-end before Gemma caption / Parakeet ASR audit",
        ],
        "note": "Mel→FSQ stub only; no 471M CleanCodec weights in Gopex",
    }
