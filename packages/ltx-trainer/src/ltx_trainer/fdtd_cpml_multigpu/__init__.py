"""Multi-GPU 3D FDTD+CPML communication study (Obieke, arXiv:2606.06910)."""

from ltx_trainer.fdtd_cpml_multigpu.config import FdtdCpmlConfig
from ltx_trainer.fdtd_cpml_multigpu.mock import evaluation_smoke
from ltx_trainer.fdtd_cpml_multigpu.paper import knowledge_bundle, paper_card
from ltx_trainer.fdtd_cpml_multigpu.pipeline import run_demo

__all__ = [
    "FdtdCpmlConfig",
    "evaluation_smoke",
    "knowledge_bundle",
    "paper_card",
    "run_demo",
]
