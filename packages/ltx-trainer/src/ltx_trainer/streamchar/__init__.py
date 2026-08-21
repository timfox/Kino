"""StreamChar: long-horizon streaming character audio-video (Tian et al., arXiv:2605.25659).

Reference: flow matching, PAP, orchestrator conditioning, paper Tables 1–2.
Full WAN/Qwen training and H100 deployment are external.
"""

from ltx_trainer.streamchar.config import StreamCharConfig
from ltx_trainer.streamchar.dit import StreamCharDiT
from ltx_trainer.streamchar.flow import corrupt_latents, flow_matching_loss
from ltx_trainer.streamchar.orchestrator import OrchestratorConditionHead
from ltx_trainer.streamchar.pap import ProgressAwarePointer
from ltx_trainer.streamchar.pipeline import (
    evaluation_demo,
    framework_card,
    table_emtd_short_clip,
    table_long_horizon,
    training_step_demo,
)

__all__ = [
    "OrchestratorConditionHead",
    "ProgressAwarePointer",
    "StreamCharConfig",
    "StreamCharDiT",
    "corrupt_latents",
    "evaluation_demo",
    "flow_matching_loss",
    "framework_card",
    "table_emtd_short_clip",
    "table_long_horizon",
    "training_step_demo",
]
