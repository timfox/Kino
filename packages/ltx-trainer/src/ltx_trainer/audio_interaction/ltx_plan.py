"""LTX / GOPEX integration."""

from __future__ import annotations

from typing import Any

from ltx_trainer.audio_interaction.config import AudioInteractionConfig


def ltx_integration_plan(cfg: AudioInteractionConfig | None = None) -> dict[str, Any]:
    cfg = cfg or AudioInteractionConfig()
    return {
        "package": "ltx_trainer.audio_interaction",
        "paper": cfg.paper_arxiv,
        "use_cases": [
            "StreamAudio-2M-style caption QA on LTX shot audio (streaming instr. follow)",
            "Proactive intervention cues on render audio (glass break, alarm) before reshoot",
            "Real-time ASR/translation sidecars during long AV-fold clips",
        ],
        "validation": "./scripts/gopex-audio-interaction.sh smoke",
        "upstream": cfg.github,
        "dataset": f"https://huggingface.co/datasets/{cfg.dataset_hub}",
    }
