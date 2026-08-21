"""RFDT-Channel RGB-LiDAR RF digital twin (arXiv:2606.01261)."""

from ltx_trainer.rfdt_channel.config import RfdtChannelConfig
from ltx_trainer.rfdt_channel.mock import evaluation_smoke
from ltx_trainer.rfdt_channel.pipeline import evaluation_demo, framework_card
from ltx_trainer.rfdt_channel.ltx_plan import ltx_integration_plan

__all__ = [
    "RfdtChannelConfig",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "ltx_integration_plan",
]
