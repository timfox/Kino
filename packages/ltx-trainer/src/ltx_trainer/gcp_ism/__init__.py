"""GCP-ISM Gauss-circle high-dimensional image-source RIRs (arXiv:2606.04358)."""

from ltx_trainer.gcp_ism.config import GCPIsmConfig
from ltx_trainer.gcp_ism.mock import evaluation_smoke
from ltx_trainer.gcp_ism.pipeline import evaluation_demo, framework_card
from ltx_trainer.gcp_ism.ltx_plan import ltx_integration_plan

__all__ = [
    "GCPIsmConfig",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
    "ltx_integration_plan",
]
