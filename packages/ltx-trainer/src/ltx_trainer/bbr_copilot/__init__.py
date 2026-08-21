"""BBR-Copilot live streaming congestion control — arXiv:2606.03468."""

from ltx_trainer.bbr_copilot.config import BBRCopilotConfig, LiveStreamConfig, MahimahiTestbedConfig, ProbeBWScenario
from ltx_trainer.bbr_copilot.pipeline import evaluation_demo, evaluation_smoke, framework_card

__all__ = [
    "BBRCopilotConfig",
    "LiveStreamConfig",
    "MahimahiTestbedConfig",
    "ProbeBWScenario",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
]
