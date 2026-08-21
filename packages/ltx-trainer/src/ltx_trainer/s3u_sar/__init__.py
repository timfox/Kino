"""S³U-SAR semantic scattering structure for SAR aircraft (Yin et al., arXiv:2606.06847)."""

from ltx_trainer.s3u_sar.config import S3USarConfig
from ltx_trainer.s3u_sar.mock import evaluation_smoke
from ltx_trainer.s3u_sar.paper import knowledge_bundle, paper_card
from ltx_trainer.s3u_sar.pipeline import run_demo

__all__ = [
    "S3USarConfig",
    "evaluation_smoke",
    "knowledge_bundle",
    "paper_card",
    "run_demo",
]
