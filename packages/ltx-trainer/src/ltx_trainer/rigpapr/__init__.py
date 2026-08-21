"""RigPAPR: rig-based animation of static PAPR clouds (Peng et al. arXiv:2606.06685)."""

from ltx_trainer.rigpapr.benchmarks import benchmarks_bundle
from ltx_trainer.rigpapr.config import RigPAPRConfig
from ltx_trainer.rigpapr.pipeline import evaluation_demo, evaluation_smoke, framework_card

__all__ = [
    "RigPAPRConfig",
    "benchmarks_bundle",
    "evaluation_demo",
    "evaluation_smoke",
    "framework_card",
]
