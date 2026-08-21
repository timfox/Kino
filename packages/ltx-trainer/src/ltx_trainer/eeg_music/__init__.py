"""Channel-oriented EEG-to-music reconstruction (arXiv:2606.04040)."""

from ltx_trainer.eeg_music.config import EegMusicConfig
from ltx_trainer.eeg_music.mock import evaluation_smoke
from ltx_trainer.eeg_music.pipeline import evaluation_demo, framework_card
from ltx_trainer.eeg_music.ltx_plan import ltx_integration_plan

__all__ = [
    "EegMusicConfig",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "ltx_integration_plan",
]
