"""Neuronal Stochastic Attention Circuit (NSAC) — probabilistic CT attention (Razzaq & Zhao, arXiv:2605.26061).

Reference implementation: OU closed-form moments on attention logits, stochastic softmax (logistic-normal
weights), heteroscedastic Gaussian output, and training loss (Gaussian NLL + epistemic separation).
Upstream code: https://github.com/itxwaleedrazzaq/neuronal_stochastic_attention_circuit
"""

from ltx_trainer.nsac.config import NSACConfig
from ltx_trainer.nsac.losses import NSACTrainingLoss, epistemic_separation_loss, gaussian_nll
from ltx_trainer.nsac.module import NSACRegressor, NSACStochasticAttention
from ltx_trainer.nsac.ou import ou_mean_variance

__all__ = [
    "NSACConfig",
    "NSACRegressor",
    "NSACStochasticAttention",
    "NSACTrainingLoss",
    "epistemic_separation_loss",
    "gaussian_nll",
    "ou_mean_variance",
]
