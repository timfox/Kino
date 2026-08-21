"""Zero-knowledge verification for frontier AI training (arXiv:2606.05433)."""

from ltx_trainer.zk_training.config import ZkTrainingConfig
from ltx_trainer.zk_training.pipeline import evaluation_demo, evaluation_smoke, framework_card

__all__ = ["ZkTrainingConfig", "evaluation_demo", "evaluation_smoke", "framework_card"]
