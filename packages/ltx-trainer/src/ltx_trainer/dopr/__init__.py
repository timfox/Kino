"""Double Preconditioning (DoPr) — arXiv:2606.06418."""

from ltx_trainer.dopr.config import DoPrConfig
from ltx_trainer.dopr.pipeline import evaluation_demo, evaluation_smoke, framework_card

__all__ = ["DoPrConfig", "evaluation_demo", "evaluation_smoke", "framework_card"]
