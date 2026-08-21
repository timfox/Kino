"""WavTTS raw waveform zero-shot TTS (arXiv:2606.03455)."""

from ltx_trainer.wavtts.config import WavTTSConfig
from ltx_trainer.wavtts.mock import evaluation_smoke
from ltx_trainer.wavtts.pipeline import evaluation_demo, framework_card
from ltx_trainer.wavtts.ltx_plan import ltx_integration_plan

__all__ = [
    "WavTTSConfig",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "ltx_integration_plan",
]
